import { LitElement, html, css, nothing } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { vannaDesignTokens } from '../styles/vanna-design-tokens.js';

export interface FieldMetadataItem {
  database: string;
  table: string;
  column: string;
  data_type: string;
  ddl_comment: string | null;
  ai_inferred: string | null;
  human_defined: string | null;
  enum_values: Record<string, string> | null;
  confidence: number;
  quality: 'high' | 'needs_confirm' | 'unknown';
  needs_review: boolean;
}

export interface DDLQualityReport {
  datasource_id: string;
  databases: string[];
  total_tables: number;
  total_fields: number;
  high_quality_fields: number;
  needs_confirm_fields: number;
  unknown_fields: number;
  high_quality_list: FieldMetadataItem[];
  needs_confirm_list: FieldMetadataItem[];
  unknown_list: FieldMetadataItem[];
}

@customElement('ddl-quality-card')
export class DDLQualityCard extends LitElement {
  static styles = [
    vannaDesignTokens,
    css`
      :host {
        display: block;
        margin-bottom: var(--vanna-space-4);
        font-family: var(--vanna-font-family-default);
      }

      .card {
        border: 1px solid var(--vanna-outline-default);
        border-radius: var(--vanna-border-radius-lg);
        background: var(--vanna-background-default);
        box-shadow: var(--vanna-shadow-sm);
        overflow: hidden;
      }

      .card-header {
        display: flex;
        align-items: center;
        padding: var(--vanna-space-4) var(--vanna-space-5);
        background: var(--vanna-background-higher);
        border-bottom: 1px solid var(--vanna-outline-default);
        gap: var(--vanna-space-3);
      }

      .card-icon {
        font-size: 1.5rem;
      }

      .card-title-section {
        flex: 1;
      }

      .card-title {
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
        color: var(--vanna-foreground-default);
      }

      .card-subtitle {
        margin: var(--vanna-space-1) 0 0 0;
        font-size: 0.875rem;
        color: var(--vanna-foreground-dimmer);
      }

      /* Statistics bar */
      .stats-bar {
        display: flex;
        gap: var(--vanna-space-4);
        padding: var(--vanna-space-3) var(--vanna-space-5);
        background: var(--vanna-background-root);
        border-bottom: 1px solid var(--vanna-outline-default);
      }

      .stat-item {
        display: flex;
        align-items: center;
        gap: var(--vanna-space-2);
        font-size: 0.875rem;
      }

      .stat-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 24px;
        height: 24px;
        padding: 0 var(--vanna-space-2);
        border-radius: var(--vanna-border-radius-md);
        font-size: 0.75rem;
        font-weight: 600;
      }

      .stat-badge.high {
        background: #d4edda;
        color: #155724;
      }

      .stat-badge.confirm {
        background: #fff3cd;
        color: #856404;
      }

      .stat-badge.unknown {
        background: #f8d7da;
        color: #721c24;
      }

      /* Section tabs */
      .section-tabs {
        display: flex;
        border-bottom: 1px solid var(--vanna-outline-default);
      }

      .section-tab {
        flex: 1;
        padding: var(--vanna-space-3) var(--vanna-space-4);
        background: none;
        border: none;
        border-bottom: 2px solid transparent;
        cursor: pointer;
        font-size: 0.875rem;
        color: var(--vanna-foreground-dimmer);
        transition: all var(--vanna-duration-200) ease;
      }

      .section-tab:hover {
        background: var(--vanna-background-higher);
      }

      .section-tab.active {
        color: var(--vanna-accent-primary-default);
        border-bottom-color: var(--vanna-accent-primary-default);
        font-weight: 600;
      }

      /* Fields list */
      .fields-section {
        max-height: 400px;
        overflow-y: auto;
      }

      .field-group {
        border-bottom: 1px solid var(--vanna-outline-default);
      }

      .field-group:last-child {
        border-bottom: none;
      }

      .field-group-header {
        display: flex;
        align-items: center;
        padding: var(--vanna-space-2) var(--vanna-space-5);
        background: var(--vanna-background-higher);
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--vanna-foreground-dimmer);
        text-transform: uppercase;
        cursor: pointer;
      }

      .field-group-header:hover {
        background: var(--vanna-background-root);
      }

      .field-group-toggle {
        margin-right: var(--vanna-space-2);
        font-size: 0.625rem;
      }

      .field-item {
        display: grid;
        grid-template-columns: 1fr 1.5fr 2fr auto;
        gap: var(--vanna-space-3);
        align-items: center;
        padding: var(--vanna-space-3) var(--vanna-space-5);
        border-bottom: 1px solid var(--vanna-outline-default);
        font-size: 0.875rem;
      }

      .field-item:last-child {
        border-bottom: none;
      }

      .field-item:hover {
        background: var(--vanna-background-root);
      }

      .field-name {
        font-family: monospace;
        font-weight: 500;
        color: var(--vanna-foreground-default);
      }

      .field-type {
        font-family: monospace;
        font-size: 0.75rem;
        color: var(--vanna-foreground-dimmer);
      }

      .field-description {
        display: flex;
        flex-direction: column;
        gap: var(--vanna-space-1);
      }

      .field-desc-source {
        font-size: 0.625rem;
        color: var(--vanna-foreground-dimmer);
        text-transform: uppercase;
      }

      .field-desc-input {
        width: 100%;
        padding: var(--vanna-space-2);
        border: 1px solid var(--vanna-outline-default);
        border-radius: var(--vanna-border-radius-sm);
        font-size: 0.875rem;
        background: var(--vanna-background-default);
        color: var(--vanna-foreground-default);
      }

      .field-desc-input:focus {
        outline: none;
        border-color: var(--vanna-accent-primary-default);
      }

      .field-actions {
        display: flex;
        gap: var(--vanna-space-2);
      }

      .btn-icon {
        width: 28px;
        height: 28px;
        padding: 0;
        border: 1px solid var(--vanna-outline-default);
        border-radius: var(--vanna-border-radius-sm);
        background: var(--vanna-background-default);
        color: var(--vanna-foreground-dimmer);
        cursor: pointer;
        font-size: 0.875rem;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all var(--vanna-duration-200) ease;
      }

      .btn-icon:hover {
        background: var(--vanna-background-higher);
        color: var(--vanna-foreground-default);
      }

      .btn-icon.confirm {
        color: #28a745;
        border-color: #28a745;
      }

      .btn-icon.confirm:hover {
        background: #28a745;
        color: white;
      }

      /* Actions footer */
      .card-actions {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--vanna-space-3) var(--vanna-space-5);
        background: var(--vanna-background-root);
        border-top: 1px solid var(--vanna-outline-default);
      }

      .actions-left {
        display: flex;
        gap: var(--vanna-space-2);
      }

      .actions-right {
        display: flex;
        gap: var(--vanna-space-2);
      }

      .btn {
        padding: var(--vanna-space-2) var(--vanna-space-4);
        border-radius: var(--vanna-border-radius-md);
        border: 1px solid var(--vanna-outline-default);
        background: var(--vanna-background-default);
        color: var(--vanna-foreground-default);
        cursor: pointer;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all var(--vanna-duration-200) ease;
      }

      .btn:hover {
        background: var(--vanna-background-higher);
      }

      .btn.primary {
        background: var(--vanna-accent-primary-default);
        color: white;
        border-color: var(--vanna-accent-primary-default);
      }

      .btn.primary:hover {
        background: var(--vanna-accent-primary-stronger);
      }

      .btn.secondary {
        background: transparent;
        color: var(--vanna-foreground-dimmer);
      }

      .empty-state {
        padding: var(--vanna-space-6);
        text-align: center;
        color: var(--vanna-foreground-dimmer);
      }

      .loading {
        padding: var(--vanna-space-6);
        text-align: center;
      }

      .spinner {
        display: inline-block;
        width: 24px;
        height: 24px;
        border: 2px solid var(--vanna-outline-default);
        border-top-color: var(--vanna-accent-primary-default);
        border-radius: 50%;
        animation: spin 1s linear infinite;
      }

      @keyframes spin {
        to { transform: rotate(360deg); }
      }
    `
  ];

  @property({ type: Object }) report: DDLQualityReport | null = null;
  @property({ type: Boolean }) loading = false;
  @property() datasourceId = '';

  @state() private activeTab: 'needs_confirm' | 'unknown' | 'high' = 'needs_confirm';
  @state() private editedFields: Map<string, string> = new Map();
  @state() private collapsedGroups: Set<string> = new Set();

  private _getFieldKey(field: FieldMetadataItem): string {
    return `${field.database}:${field.table}:${field.column}`;
  }

  private _groupFieldsByTable(fields: FieldMetadataItem[]): Map<string, FieldMetadataItem[]> {
    const groups = new Map<string, FieldMetadataItem[]>();
    for (const field of fields) {
      const key = `${field.database}.${field.table}`;
      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key)!.push(field);
    }
    return groups;
  }

  private _handleFieldEdit(field: FieldMetadataItem, value: string) {
    const key = this._getFieldKey(field);
    this.editedFields.set(key, value);
    this.requestUpdate();
  }

  private _getFieldValue(field: FieldMetadataItem): string {
    const key = this._getFieldKey(field);
    if (this.editedFields.has(key)) {
      return this.editedFields.get(key)!;
    }
    return field.human_defined || field.ai_inferred || field.ddl_comment || '';
  }

  private _toggleGroup(groupKey: string) {
    if (this.collapsedGroups.has(groupKey)) {
      this.collapsedGroups.delete(groupKey);
    } else {
      this.collapsedGroups.add(groupKey);
    }
    this.requestUpdate();
  }

  private _confirmField(field: FieldMetadataItem) {
    const key = this._getFieldKey(field);
    const currentValue = this._getFieldValue(field);
    if (currentValue) {
      this.editedFields.set(key, currentValue);
      this.requestUpdate();
    }
  }

  private _confirmAll() {
    const fields = this._getCurrentFields();
    for (const field of fields) {
      const value = this._getFieldValue(field);
      if (value) {
        this.editedFields.set(this._getFieldKey(field), value);
      }
    }
    this.requestUpdate();
  }

  private _getCurrentFields(): FieldMetadataItem[] {
    if (!this.report) return [];
    switch (this.activeTab) {
      case 'needs_confirm':
        return this.report.needs_confirm_list;
      case 'unknown':
        return this.report.unknown_list;
      case 'high':
        return this.report.high_quality_list;
      default:
        return [];
    }
  }

  private _handleSaveAndGenerate() {
    // Collect all edited/confirmed fields
    const fieldsToSave: FieldMetadataItem[] = [];
    
    // Include all fields from needs_confirm and unknown
    const allFields = [
      ...(this.report?.needs_confirm_list || []),
      ...(this.report?.unknown_list || [])
    ];

    for (const field of allFields) {
      const key = this._getFieldKey(field);
      const editedValue = this.editedFields.get(key);
      
      if (editedValue !== undefined) {
        fieldsToSave.push({
          ...field,
          human_defined: editedValue
        });
      }
    }

    this.dispatchEvent(new CustomEvent('save-and-generate', {
      detail: {
        datasourceId: this.datasourceId,
        fields: fieldsToSave
      },
      bubbles: true,
      composed: true
    }));
  }

  private _handleSkipGenerate() {
    this.dispatchEvent(new CustomEvent('skip-generate', {
      detail: {
        datasourceId: this.datasourceId
      },
      bubbles: true,
      composed: true
    }));
  }

  private _renderFieldItem(field: FieldMetadataItem) {
    const key = this._getFieldKey(field);
    const isEdited = this.editedFields.has(key);
    const currentValue = this._getFieldValue(field);
    const source = field.human_defined ? 'human' : (field.ai_inferred ? 'ai' : 'ddl');

    return html`
      <div class="field-item">
        <div>
          <div class="field-name">${field.column}</div>
          <div class="field-type">${field.data_type}</div>
        </div>
        <div class="field-description">
          <span class="field-desc-source">
            ${source === 'human' ? '人工定义' : (source === 'ai' ? 'AI推断' : 'DDL注释')}
            ${isEdited ? ' (已修改)' : ''}
          </span>
          <input 
            type="text"
            class="field-desc-input"
            .value=${currentValue}
            placeholder="请输入字段描述..."
            @input=${(e: Event) => this._handleFieldEdit(field, (e.target as HTMLInputElement).value)}
          />
        </div>
        <div class="field-actions">
          ${field.quality !== 'high' ? html`
            <button 
              class="btn-icon confirm" 
              title="确认"
              @click=${() => this._confirmField(field)}
            >
              ✓
            </button>
          ` : nothing}
        </div>
      </div>
    `;
  }

  private _renderFieldsSection() {
    const fields = this._getCurrentFields();
    
    if (fields.length === 0) {
      return html`
        <div class="empty-state">
          ${this.activeTab === 'high' ? '没有高质量字段' : 
            this.activeTab === 'needs_confirm' ? '没有需要确认的字段' : 
            '没有未知字段'}
        </div>
      `;
    }

    const groups = this._groupFieldsByTable(fields);

    return html`
      <div class="fields-section">
        ${Array.from(groups.entries()).map(([tableName, tableFields]) => {
          const isCollapsed = this.collapsedGroups.has(tableName);
          return html`
            <div class="field-group">
              <div 
                class="field-group-header"
                @click=${() => this._toggleGroup(tableName)}
              >
                <span class="field-group-toggle">${isCollapsed ? '▶' : '▼'}</span>
                ${tableName} (${tableFields.length} 个字段)
              </div>
              ${!isCollapsed ? tableFields.map(field => this._renderFieldItem(field)) : nothing}
            </div>
          `;
        })}
      </div>
    `;
  }

  render() {
    if (this.loading) {
      return html`
        <div class="card">
          <div class="loading">
            <div class="spinner"></div>
            <p>正在分析 DDL 质量...</p>
          </div>
        </div>
      `;
    }

    if (!this.report) {
      return html`
        <div class="card">
          <div class="empty-state">
            <p>暂无质量分析数据</p>
          </div>
        </div>
      `;
    }

    const { 
      total_tables, 
      total_fields, 
      high_quality_fields, 
      needs_confirm_fields, 
      unknown_fields 
    } = this.report;

    return html`
      <div class="card">
        <div class="card-header">
          <span class="card-icon">📊</span>
          <div class="card-title-section">
            <h3 class="card-title">DDL 质量分析报告</h3>
            <p class="card-subtitle">
              共 ${total_tables} 个表，${total_fields} 个字段
            </p>
          </div>
        </div>

        <div class="stats-bar">
          <div class="stat-item">
            <span class="stat-badge high">${high_quality_fields}</span>
            <span>高质量</span>
          </div>
          <div class="stat-item">
            <span class="stat-badge confirm">${needs_confirm_fields}</span>
            <span>需确认</span>
          </div>
          <div class="stat-item">
            <span class="stat-badge unknown">${unknown_fields}</span>
            <span>需补充</span>
          </div>
        </div>

        <div class="section-tabs">
          <button 
            class="section-tab ${this.activeTab === 'needs_confirm' ? 'active' : ''}"
            @click=${() => this.activeTab = 'needs_confirm'}
          >
            需确认 (${needs_confirm_fields})
          </button>
          <button 
            class="section-tab ${this.activeTab === 'unknown' ? 'active' : ''}"
            @click=${() => this.activeTab = 'unknown'}
          >
            需补充 (${unknown_fields})
          </button>
          <button 
            class="section-tab ${this.activeTab === 'high' ? 'active' : ''}"
            @click=${() => this.activeTab = 'high'}
          >
            高质量 (${high_quality_fields})
          </button>
        </div>

        ${this._renderFieldsSection()}

        <div class="card-actions">
          <div class="actions-left">
            <button class="btn secondary" @click=${this._confirmAll}>
              全部确认
            </button>
          </div>
          <div class="actions-right">
            <button class="btn secondary" @click=${this._handleSkipGenerate}>
              跳过，直接生成
            </button>
            <button class="btn primary" @click=${this._handleSaveAndGenerate}>
              保存并生成语料
            </button>
          </div>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'ddl-quality-card': DDLQualityCard;
  }
}
