"""
智能语料生成模块
从数据源提取DDL，使用AI生成字段文档和SQL示例
"""
import os
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from datasource_manager import get_datasource_manager


class CorpusGenerator:
    """语料生成器"""
    
    def __init__(self, vn_instance):
        """
        初始化语料生成器
        
        Args:
            vn_instance: Vanna实例，用于调用AI生成内容
        """
        self.vn = vn_instance
        self.ds_manager = get_datasource_manager()
        self._corpus_timeout_seconds = self._get_corpus_timeout_seconds()

    def _get_corpus_timeout_seconds(self) -> int:
        """
        获取语料生成超时时间（秒）
        """
        try:
            return int(os.getenv("ZHIPU_YULIAO_TIMEOUT", "180"))
        except ValueError:
            return 180

    def _submit_prompt_with_timeout(self, message_log, model: str, context: str) -> str:
        """
        调用 LLM 并设置超时，避免请求长期阻塞
        """
        timeout_seconds = self._corpus_timeout_seconds
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.vn.submit_prompt, message_log, model=model)
            try:
                return future.result(timeout=timeout_seconds)
            except FutureTimeoutError:
                raise TimeoutError(
                    f"{context} 调用超时（{timeout_seconds}s），请检查模型服务或缩小表数量"
                )
    
    def generate_corpus_from_datasource(
        self, 
        datasource_id: str,
        databases: Optional[List[str]] = None,
        generate_documentation: bool = True,
        generate_sql_examples: bool = True
    ) -> Dict[str, Any]:
        """
        从数据源生成语料
        
        Args:
            datasource_id: 数据源ID
            databases: 要处理的数据库列表，如果不提供则使用数据源配置的databases
            generate_documentation: 是否生成字段文档
            generate_sql_examples: 是否生成SQL示例
            
        Returns:
            {
                "datasource_name": str,
                "table_count": int,
                "corpus": List[Dict]  # 语料列表
            }
        """
        ds = self.ds_manager.get(datasource_id)
        if not ds:
            raise ValueError(f"数据源 {datasource_id} 不存在")
        
        # 确定要处理的数据库
        target_databases = databases or ds.databases
        if not target_databases:
            raise ValueError("数据源未选择任何数据库，请先在数据源管理中配置")
        
        corpus = []
        total_tables = 0
        
        # 第一步：收集所有需要处理的表
        tables_by_db = {}  # {db_name: [table_info, ...]}
        for db_name in target_databases:
            try:
                print(f"[CorpusGenerator] 开始处理数据库: {db_name}")
                tables = self.ds_manager.get_tables(datasource_id, db_name)
                print(f"[CorpusGenerator] 数据库 {db_name} 找到 {len(tables)} 个表")
                
                if not tables:
                    print(f"[CorpusGenerator] 警告: 数据库 {db_name} 没有找到任何表")
                    continue
                
                total_tables += len(tables)
                tables_by_db[db_name] = tables
            except Exception as e:
                print(f"[CorpusGenerator] 处理数据库 {db_name} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not tables_by_db:
            print(f"[CorpusGenerator] 没有找到任何表，返回空结果")
            return {
                "datasource_name": ds.name,
                "table_count": 0,
                "corpus": []
            }
        
        # 第二步：并发获取所有表的DDL（这部分很快，可以高并发）
        print(f"[CorpusGenerator] 开始并发获取所有表的DDL...")
        all_ddls = {}  # {(db_name, table_name): ddl}
        
        # 收集所有表
        all_tables = []
        for db_name, tables in tables_by_db.items():
            for table_info in tables:
                all_tables.append((db_name, table_info['name']))
        
        # 高并发获取DDL（数据库查询很快，可以设置更高的并发数）
        ddl_workers = min(10, len(all_tables))
        with ThreadPoolExecutor(max_workers=ddl_workers) as executor:
            future_to_table = {
                executor.submit(
                    self._get_table_ddl,
                    datasource_id,
                    db_name,
                    table_name
                ): (db_name, table_name)
                for db_name, table_name in all_tables
            }
            
            completed = 0
            ddl_timeout_seconds = self._get_corpus_ddl_timeout_seconds()
            ddl_timeout_total = max(ddl_timeout_seconds, ddl_timeout_seconds * max(1, len(all_tables)))
            try:
                for future in as_completed(future_to_table, timeout=ddl_timeout_total):
                    db_name, table_name = future_to_table[future]
                    completed += 1
                    try:
                        ddl = future.result()
                        if ddl:
                            all_ddls[(db_name, table_name)] = ddl
                            print(f"[CorpusGenerator] [{completed}/{len(all_tables)}] 获取DDL: {db_name}.{table_name}")
                        else:
                            print(f"[CorpusGenerator] [{completed}/{len(all_tables)}] 警告: {db_name}.{table_name} DDL为空")
                    except Exception as e:
                        print(f"[CorpusGenerator] [{completed}/{len(all_tables)}] 获取DDL失败 {db_name}.{table_name}: {e}")
                        continue
            except FutureTimeoutError:
                pending = [f for f in future_to_table if not f.done()]
                for f in pending:
                    f.cancel()
                print(f"[CorpusGenerator] 获取DDL超时，未完成表数量: {len(pending)}")
        
        print(f"[CorpusGenerator] DDL获取完成，共 {len(all_ddls)} 个表")
        
        # 第三步：按数据库分组，批量生成文档和SQL示例
        for db_name, tables in tables_by_db.items():
            # 收集该数据库下所有表的DDL
            db_ddls = {}
            for table_info in tables:
                table_name = table_info['name']
                if (db_name, table_name) in all_ddls:
                    db_ddls[table_name] = all_ddls[(db_name, table_name)]
            
            if not db_ddls:
                print(f"[CorpusGenerator] 数据库 {db_name} 没有有效的DDL，跳过")
                continue
            
            # 添加DDL语料
            for table_name, ddl in db_ddls.items():
                corpus.append({
                    "id": f"ddl-{db_name}-{table_name}",
                    "type": "ddl",
                    "content": f"-- {db_name}.{table_name}\n{ddl}",
                    "question": None,
                    "tags": f"{table_name}表DDL,{db_name}"
                })
            
            # 批量生成字段文档（如果表数量>10，分批处理）
            if generate_documentation:
                print(f"[CorpusGenerator] 批量生成字段文档: {db_name} ({len(db_ddls)} 个表)")
                docs = self._generate_batch_documentation_with_chunks(db_ddls, db_name)
                corpus.extend(docs)
            
            # 批量生成SQL示例（如果表数量>10，分批处理）
            if generate_sql_examples:
                print(f"[CorpusGenerator] 批量生成SQL示例: {db_name} ({len(db_ddls)} 个表)")
                sql_examples = self._generate_batch_sql_examples_with_chunks(db_ddls, db_name)
                corpus.extend(sql_examples)
        
        print(f"[CorpusGenerator] 所有表处理完成: 表数量={total_tables}, 语料数量={len(corpus)}")
        return {
            "datasource_name": ds.name,
            "table_count": total_tables,
            "corpus": corpus
        }
    
    def _get_table_ddl(self, datasource_id: str, db_name: str, table_name: str) -> Optional[str]:
        """
        获取单个表的DDL（用于并发调用）
        
        Returns:
            DDL字符串，如果失败返回None
        """
        try:
            ddl = self.ds_manager.get_table_ddl(datasource_id, db_name, table_name)
            return ddl if ddl else None
        except Exception as e:
            print(f"[CorpusGenerator] 获取DDL失败 ({db_name}.{table_name}): {e}")
            return None

    def _get_corpus_ddl_timeout_seconds(self) -> int:
        """
        获取DDL拉取超时时间（秒）
        """
        try:
            return int(os.getenv("CORPUS_DDL_TIMEOUT", "30"))
        except ValueError:
            return 30
    
    def _split_dict_into_chunks(self, d: Dict[str, Any], chunk_size: int = 10) -> List[Dict[str, Any]]:
        """
        将字典按指定大小分批
        
        Args:
            d: 要分批的字典
            chunk_size: 每批的大小
            
        Returns:
            字典列表，每个字典是一批
        """
        items = list(d.items())
        chunks = []
        for i in range(0, len(items), chunk_size):
            chunk = dict(items[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def _generate_batch_documentation_with_chunks(
        self,
        table_ddls: Dict[str, str],
        database: str,
        chunk_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        批量生成文档，如果表数量超过chunk_size则分批处理
        
        Args:
            table_ddls: {table_name: ddl} 字典
            database: 数据库名
            chunk_size: 每批处理的表数量，默认10
            
        Returns:
            文档语料列表
        """
        if len(table_ddls) <= chunk_size:
            # 表数量不超过限制，直接批量处理
            return self._generate_batch_documentation(table_ddls, database)
        
        # 表数量超过限制，分批处理
        print(f"[CorpusGenerator] 表数量({len(table_ddls)})超过{chunk_size}，分批处理")
        chunks = self._split_dict_into_chunks(table_ddls, chunk_size)
        all_docs = []
        
        for idx, chunk in enumerate(chunks, 1):
            print(f"[CorpusGenerator] 处理第 {idx}/{len(chunks)} 批文档生成 ({len(chunk)} 个表)")
            docs = self._generate_batch_documentation(chunk, database)
            all_docs.extend(docs)
        
        print(f"[CorpusGenerator] 所有批次文档生成完成: 共 {len(all_docs)} 条")
        return all_docs
    
    def _generate_batch_documentation(
        self, 
        table_ddls: Dict[str, str], 
        database: str
    ) -> List[Dict[str, Any]]:
        """
        批量生成多个表的字段文档（一次性调用AI）
        
        Args:
            table_ddls: {table_name: ddl} 字典
            database: 数据库名
            
        Returns:
            文档语料列表
        """
        if not table_ddls:
            return []
        
        # 构建批量prompt
        ddl_text = "\n\n".join([
            f"-- 表名: {database}.{table_name}\n```sql\n{ddl}\n```"
            for table_name, ddl in table_ddls.items()
        ])
        
        prompt_text = f"""你是一个数据库文档专家。请根据以下多个表的DDL，为每个表生成详细的字段说明文档。

{ddl_text}

要求：
1. 为每个表分别生成文档
2. 推断每个字段的业务含义（根据字段名和类型）
3. 对于状态字段（status, state等），列出可能的取值和含义
4. 对于时间字段，说明格式（时间戳/datetime/date）
5. 对于外键字段，说明关联关系
6. 使用Markdown格式，每个表使用二级标题（## {database}.表名）
7. 文档结构：
   - 表名和用途说明
   - 字段列表（每个字段一行，包含：字段名、类型、说明）
   - 重要字段的详细说明（状态码、时间格式等）
   - 索引和约束说明（如果有）

输出格式：为每个表生成一个独立的文档块，用 ## 标题分隔。

只输出文档内容，不要输出其他解释文字。"""
        
        try:
            message_log = [self.vn.user_message(prompt_text)]
            corpus_model = os.getenv('ZHIPU_YULIAO_MODEL', 'GLM-4.5')
            print(f"[CorpusGenerator] 使用模型 {corpus_model} 批量生成字段文档: {database} ({len(table_ddls)} 个表)")
            response = self._submit_prompt_with_timeout(
                message_log, model=corpus_model, context=f"批量生成字段文档({database})"
            )
            
            if not isinstance(response, str):
                response = str(response)
            response = response.strip()
            
            # 按表分割文档（根据 ## 标题）
            docs = []
            current_table = None
            current_content = []
            
            for line in response.split('\n'):
                if line.startswith('##'):
                    # 保存上一个表的文档
                    if current_table and current_content:
                        docs.append({
                            "table": current_table,
                            "content": '\n'.join(current_content)
                        })
                    # 开始新表
                    # 尝试从标题中提取表名
                    table_match = re.search(r'##\s+(?:[\w.]+\.)?(\w+)', line)
                    if table_match:
                        current_table = table_match.group(1)
                    else:
                        current_table = None
                    current_content = [line]
                elif current_table:
                    current_content.append(line)
            
            # 保存最后一个表的文档
            if current_table and current_content:
                docs.append({
                    "table": current_table,
                    "content": '\n'.join(current_content)
                })
            
            # 如果无法分割，尝试为每个表生成单独的文档
            if not docs:
                print(f"[CorpusGenerator] 无法分割批量文档，回退到单表生成模式")
                return self._generate_individual_documentations(table_ddls, database)
            
            # 转换为语料格式
            result = []
            for doc in docs:
                table_name = doc["table"]
                if table_name in table_ddls:  # 确保表名匹配
                    result.append({
                        "id": f"doc-{database}-{table_name}",
                        "type": "documentation",
                        "content": doc["content"],
                        "question": f"{table_name}表有哪些字段",
                        "tags": f"字段说明,{table_name},{database}"
                    })
            
            print(f"[CorpusGenerator] 批量生成文档完成: {len(result)}/{len(table_ddls)} 个表")
            return result
            
        except TimeoutError as e:
            print(f"[CorpusGenerator] 批量生成字段文档超时 ({database}): {e}")
            return []
        except Exception as e:
            print(f"[CorpusGenerator] 批量生成字段文档失败 ({database}): {e}")
            import traceback
            traceback.print_exc()
            # 回退到单表生成模式
            return self._generate_individual_documentations(table_ddls, database)
    
    def _generate_individual_documentations(
        self,
        table_ddls: Dict[str, str],
        database: str
    ) -> List[Dict[str, Any]]:
        """
        为每个表单独生成文档（批量失败时的回退方案）
        """
        result = []
        for table_name, ddl in table_ddls.items():
            doc_content = self._generate_field_documentation(ddl, database, table_name)
            if doc_content:
                result.append({
                    "id": f"doc-{database}-{table_name}",
                    "type": "documentation",
                    "content": doc_content,
                    "question": f"{table_name}表有哪些字段",
                    "tags": f"字段说明,{table_name},{database}"
                })
        return result
    
    def _generate_batch_sql_examples_with_chunks(
        self,
        table_ddls: Dict[str, str],
        database: str,
        chunk_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        批量生成SQL示例，如果表数量超过chunk_size则分批处理
        
        Args:
            table_ddls: {table_name: ddl} 字典
            database: 数据库名
            chunk_size: 每批处理的表数量，默认10
            
        Returns:
            SQL示例语料列表
        """
        if len(table_ddls) <= chunk_size:
            # 表数量不超过限制，直接批量处理
            return self._generate_batch_sql_examples(table_ddls, database)
        
        # 表数量超过限制，分批处理
        print(f"[CorpusGenerator] 表数量({len(table_ddls)})超过{chunk_size}，分批处理")
        chunks = self._split_dict_into_chunks(table_ddls, chunk_size)
        all_examples = []
        
        for idx, chunk in enumerate(chunks, 1):
            print(f"[CorpusGenerator] 处理第 {idx}/{len(chunks)} 批SQL示例生成 ({len(chunk)} 个表)")
            examples = self._generate_batch_sql_examples(chunk, database)
            all_examples.extend(examples)
        
        print(f"[CorpusGenerator] 所有批次SQL示例生成完成: 共 {len(all_examples)} 条")
        return all_examples
    
    def _generate_batch_sql_examples(
        self,
        table_ddls: Dict[str, str],
        database: str
    ) -> List[Dict[str, Any]]:
        """
        批量生成多个表的SQL示例（一次性调用AI）
        
        Args:
            table_ddls: {table_name: ddl} 字典
            database: 数据库名
            
        Returns:
            SQL示例语料列表
        """
        if not table_ddls:
            return []
        
        # 构建批量prompt
        ddl_text = "\n\n".join([
            f"-- 表名: {database}.{table_name}\n```sql\n{ddl}\n```"
            for table_name, ddl in table_ddls.items()
        ])
        
        prompt_text = f"""你是一个SQL查询专家。根据以下多个表的结构，为每个表生成5-8个常见的SQL查询示例。

{ddl_text}

要求：
- 为每个表分别生成SQL示例
- SQL查询要符合MySQL语法
- 问题要贴近实际业务场景（中文）
- 包含不同类型的查询：简单查询、条件过滤、聚合统计、排序、分页等
- 使用完整表名：{database}.表名

输出格式（JSON对象，键为表名）：
{{
  "表名1": [
    {{"question": "查询问题", "sql": "SELECT ... FROM {database}.表名1 ..."}},
    ...
  ],
  "表名2": [
    {{"question": "查询问题", "sql": "SELECT ... FROM {database}.表名2 ..."}},
    ...
  ]
}}

只输出JSON对象，不要其他文字。"""
        
        try:
            message_log = [self.vn.user_message(prompt_text)]
            corpus_model = os.getenv('ZHIPU_YULIAO_MODEL', 'GLM-4.5')
            print(f"[CorpusGenerator] 使用模型 {corpus_model} 批量生成SQL示例: {database} ({len(table_ddls)} 个表)")
            response = self._submit_prompt_with_timeout(
                message_log, model=corpus_model, context=f"批量生成SQL示例({database})"
            )
            
            if not isinstance(response, str):
                response = str(response)
            
            # 尝试提取JSON对象
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                import json
                try:
                    sql_examples_dict = json.loads(json_match.group())
                    
                    result = []
                    for table_name, examples in sql_examples_dict.items():
                        if table_name not in table_ddls:
                            continue
                        if not isinstance(examples, list):
                            continue
                        for idx, example in enumerate(examples[:8]):
                            if isinstance(example, dict) and 'question' in example and 'sql' in example:
                                result.append({
                                    "id": f"sql-{database}-{table_name}-{idx}",
                                    "type": "sql",
                                    "content": example['sql'],
                                    "question": example['question'],
                                    "tags": f"SQL示例,{table_name},{database}"
                                })
                    
                    print(f"[CorpusGenerator] 批量生成SQL示例完成: {len(result)} 条")
                    return result
                except json.JSONDecodeError as e:
                    print(f"[CorpusGenerator] JSON解析失败: {e}")
            
            # 回退到单表生成模式
            print(f"[CorpusGenerator] 无法解析批量SQL示例，回退到单表生成模式")
            return self._generate_individual_sql_examples(table_ddls, database)
            
        except TimeoutError as e:
            print(f"[CorpusGenerator] 批量生成SQL示例超时 ({database}): {e}")
            return []
        except Exception as e:
            print(f"[CorpusGenerator] 批量生成SQL示例失败 ({database}): {e}")
            import traceback
            traceback.print_exc()
            # 回退到单表生成模式
            return self._generate_individual_sql_examples(table_ddls, database)
    
    def _generate_individual_sql_examples(
        self,
        table_ddls: Dict[str, str],
        database: str
    ) -> List[Dict[str, Any]]:
        """
        为每个表单独生成SQL示例（批量失败时的回退方案）
        """
        result = []
        for table_name, ddl in table_ddls.items():
            sql_examples = self._generate_sql_examples(ddl, database, table_name)
            result.extend(sql_examples)
        return result
    
    def _generate_field_documentation(self, ddl: str, database: str, table: str) -> Optional[str]:
        """
        使用AI生成字段说明文档
        
        Args:
            ddl: CREATE TABLE语句
            database: 数据库名
            table: 表名
            
        Returns:
            Markdown格式的字段说明文档
        """
        prompt_text = f"""你是一个数据库文档专家。请根据以下DDL生成详细的字段说明文档。

```sql
{ddl}
```

要求：
1. 推断每个字段的业务含义（根据字段名和类型）
2. 对于状态字段（status, state等），列出可能的取值和含义
3. 对于时间字段，说明格式（时间戳/datetime/date）
4. 对于外键字段，说明关联关系
5. 使用Markdown格式，包含二级标题（##）和列表
6. 文档结构：
   - 表名和用途说明
   - 字段列表（每个字段一行，包含：字段名、类型、说明）
   - 重要字段的详细说明（状态码、时间格式等）
   - 索引和约束说明（如果有）

只输出文档内容，不要输出其他解释文字。"""
        
        try:
            # submit_prompt 需要消息列表格式
            message_log = [
                self.vn.user_message(prompt_text)
            ]
            
            # 使用专门的语料生成模型（避免限速）
            corpus_model = os.getenv('ZHIPU_YULIAO_MODEL', 'GLM-4.5')
            print(f"[CorpusGenerator] 使用模型 {corpus_model} 生成字段文档: {database}.{table}")
            response = self._submit_prompt_with_timeout(
                message_log, model=corpus_model, context=f"生成字段文档({database}.{table})"
            )
            
            # 确保 response 是字符串
            if isinstance(response, str):
                response = response.strip()
            else:
                # 如果返回的不是字符串，尝试转换
                response = str(response).strip()
            
            if not response.startswith('##'):
                response = f"## {database}.{table} 字段说明\n\n{response}"
            return response
        except Exception as e:
            print(f"[CorpusGenerator] 生成字段文档失败 ({database}.{table}): {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_sql_examples(self, ddl: str, database: str, table: str) -> List[Dict[str, Any]]:
        """
        使用AI生成SQL查询示例
        
        Args:
            ddl: CREATE TABLE语句
            database: 数据库名
            table: 表名
            
        Returns:
            SQL示例列表
        """
        prompt_text = f"""你是一个SQL查询专家。根据以下表结构，生成5-8个常见的SQL查询示例，每个示例包含：
1. 自然语言问题（中文）
2. 对应的SQL查询

表结构：
```sql
{ddl}
```

要求：
- SQL查询要符合MySQL语法
- 问题要贴近实际业务场景
- 包含不同类型的查询：简单查询、条件过滤、聚合统计、排序、分页等
- 使用完整表名：{database}.{table}

输出格式（JSON数组）：
[
  {{
    "question": "查询问题",
    "sql": "SELECT ... FROM {database}.{table} ..."
  }},
  ...
]

只输出JSON数组，不要其他文字。"""
        
        try:
            # submit_prompt 需要消息列表格式
            message_log = [
                self.vn.user_message(prompt_text)
            ]
            
            # 使用专门的语料生成模型（避免限速）
            corpus_model = os.getenv('ZHIPU_YULIAO_MODEL', 'GLM-4.5')
            print(f"[CorpusGenerator] 使用模型 {corpus_model} 生成SQL示例: {database}.{table}")
            response = self._submit_prompt_with_timeout(
                message_log, model=corpus_model, context=f"生成SQL示例({database}.{table})"
            )
            
            # 确保 response 是字符串
            if not isinstance(response, str):
                response = str(response)
            
            # 尝试从响应中提取JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                import json
                sql_examples = json.loads(json_match.group())
                
                result = []
                for idx, example in enumerate(sql_examples[:8]):  # 最多8个
                    if isinstance(example, dict) and 'question' in example and 'sql' in example:
                        result.append({
                            "id": f"sql-{database}-{table}-{idx}",
                            "type": "sql",
                            "content": example['sql'],
                            "question": example['question'],
                            "tags": f"SQL示例,{table},{database}"
                        })
                return result
            else:
                # 如果无法解析JSON，尝试手动生成一些基础示例
                print(f"[CorpusGenerator] 无法从响应中提取JSON，使用基础示例")
                return self._generate_basic_sql_examples(database, table)
        except Exception as e:
            print(f"[CorpusGenerator] 生成SQL示例失败 ({database}.{table}): {e}")
            import traceback
            traceback.print_exc()
            return self._generate_basic_sql_examples(database, table)
    
    def _generate_basic_sql_examples(self, database: str, table: str) -> List[Dict[str, Any]]:
        """
        生成基础的SQL示例（当AI生成失败时使用）
        """
        examples = [
            {
                "id": f"sql-{database}-{table}-1",
                "type": "sql",
                "content": f"SELECT * FROM {database}.{table} LIMIT 10",
                "question": f"查询{table}表的前10条记录",
                "tags": f"SQL示例,{table},{database}"
            },
            {
                "id": f"sql-{database}-{table}-2",
                "type": "sql",
                "content": f"SELECT COUNT(*) as total FROM {database}.{table}",
                "question": f"统计{table}表的总记录数",
                "tags": f"SQL示例,{table},{database}"
            }
        ]
        return examples
