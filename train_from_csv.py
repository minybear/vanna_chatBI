"""
增量训练脚本 - 从CSV文件批量导入知识库语料
使用方法: python train_from_csv.py
"""
import pandas as pd
from app import vn
import os

def train_from_csv(csv_file_path: str):
    """
    从CSV文件读取数据并批量训练Vanna模型
    
    CSV格式要求:
    - 类型 (type): ddl, documentation, sql
    - 内容 (content): DDL语句/文档内容/SQL语句
    - 用户问题示例 (question): SQL类型必填
    - 业务标签 (tags): 可选,用于分类
    """
    
    if not os.path.exists(csv_file_path):
        print(f"❌ 错误: CSV文件不存在: {csv_file_path}")
        return
    
    print(f"📂 正在读取CSV文件: {csv_file_path}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        print(f"✅ 成功读取 {len(df)} 条记录")
        
        # 显示CSV列名
        print(f"📋 CSV列名: {df.columns.tolist()}")
        
        # 统计训练数据类型
        if '类型' in df.columns:
            print(f"\n📊 数据类型分布:")
            type_counts = df['类型'].value_counts()
            for dtype, count in type_counts.items():
                print(f"   - {dtype}: {count} 条")
        
        # 开始批量训练
        print(f"\n🚀 开始增量训练...")
        success_count = 0
        fail_count = 0
        
        for idx, row in df.iterrows():
            try:
                data_type = row['类型']
                content = row['内容']
                question = row.get('用户问题示例', None)
                tags = row.get('业务标签', '')
                
                # 根据类型调用不同的训练方法
                if data_type == 'ddl':
                    print(f"[{idx+1}/{len(df)}] 训练DDL: {content[:50]}...")
                    vn.train(ddl=content)
                    
                elif data_type == 'documentation':
                    print(f"[{idx+1}/{len(df)}] 训练文档: {content[:50]}... (标签: {tags})")
                    vn.train(documentation=content)
                    
                elif data_type == 'sql':
                    if pd.isna(question) or not question:
                        print(f"⚠️  [{idx+1}/{len(df)}] 跳过: SQL类型缺少问题示例")
                        fail_count += 1
                        continue
                    print(f"[{idx+1}/{len(df)}] 训练SQL: {question} -> {content[:50]}...")
                    vn.train(question=question, sql=content)
                    
                else:
                    print(f"⚠️  [{idx+1}/{len(df)}] 跳过: 未知类型 '{data_type}'")
                    fail_count += 1
                    continue
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ [{idx+1}/{len(df)}] 训练失败: {str(e)}")
                fail_count += 1
        
        # 输出训练结果
        print(f"\n" + "="*60)
        print(f"✅ 训练完成!")
        print(f"   成功: {success_count} 条")
        print(f"   失败: {fail_count} 条")
        print(f"   总计: {len(df)} 条")
        print("="*60)
        
        # 显示当前知识库统计
        print(f"\n📚 当前知识库统计:")
        training_data = vn.get_training_data()
        if training_data is not None and not training_data.empty:
            print(f"   知识库总记录数: {len(training_data)}")
            if 'training_data_type' in training_data.columns:
                type_stats = training_data['training_data_type'].value_counts()
                for dtype, count in type_stats.items():
                    print(f"   - {dtype}: {count} 条")
        else:
            print("   知识库为空")
            
    except Exception as e:
        print(f"❌ 训练过程出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # CSV文件路径 - 使用项目内的知识库文件
    csv_file = r"d:\redtea\workSpc\dev-tools\ideaProjects\ai\chatBi\vanna_chatBI\ES_RAG_Knowledge_Base_v2.csv"
    
    print("=" * 60)
    print("🤖 redtea chatBi - 增量知识库训练工具")
    print("=" * 60)
    print()
    
    train_from_csv(csv_file)
    
    print("\n✨ 训练脚本执行完毕!")














