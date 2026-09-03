/** Models admin panel logic for settings.html */

window.ModelsAdmin = {
  modelSubTab: 'paths',
  appConfig: {},
  configSections: [],
  localCatalog: [],
  cloudCatalog: [],
  installedModels: [],
  modelsDir: '',
  downloadTasks: [],
  downloadPollTimer: null,

  localFilter: '',
  localRoleFilter: '',
  cloudRegionFilter: '',

  selectedLocalModel: null,
  selectedQuant: '',
  selectedProvider: null,
  cloudSetup: { api_key: '', model_id: '', set_default: true },

  async loadAll() {
    await Promise.all([
      this.loadAppConfig(),
      this.loadLocalCatalog(),
      this.loadCloudCatalog(),
      this.loadInstalled(),
      this.loadDownloads(),
      this.loadModels(),
    ]);
  },

  async loadAppConfig() {
    const res = await fetch('/api/admin/config');
    const d = await res.json();
    this.appConfig = d.config || {};
    this.configSections = d.sections || [];
    this.modelsDir = this.appConfig?.paths?.models_dir || '';
  },

  async saveAppConfig(partial) {
    const res = await fetch('/api/admin/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: partial }),
    });
    if (res.ok) {
      await this.loadAppConfig();
      return true;
    }
    const d = await res.json();
    throw new Error(d.detail || '保存失败');
  },

  async savePaths() {
    try {
      await this.saveAppConfig({
        paths: {
          models_dir: this.modelsDir,
          cache_dir: this.appConfig?.paths?.cache_dir,
          logs_dir: this.appConfig?.paths?.logs_dir,
        },
      });
      this.toast('路径配置已保存到 config.yaml', 'success');
      await this.loadInstalled();
    } catch (e) {
      this.toast(e.message, 'error');
    }
  },

  async saveRouting() {
    try {
      await this.saveAppConfig({
        routing: this.appConfig.routing,
        mcp: this.appConfig.mcp,
        network: this.appConfig.network,
      });
      this.toast('系统配置已保存', 'success');
    } catch (e) {
      this.toast(e.message, 'error');
    }
  },

  async loadLocalCatalog() {
    const res = await fetch('/api/admin/models/catalog/local');
    const d = await res.json();
    this.localCatalog = d.models || [];
  },

  async loadCloudCatalog() {
    const res = await fetch('/api/admin/models/catalog/cloud');
    const d = await res.json();
    this.cloudCatalog = d.providers || [];
  },

  async loadInstalled() {
    const res = await fetch('/api/admin/models/installed');
    const d = await res.json();
    this.installedModels = d.models || [];
    if (d.models_dir) this.modelsDir = d.models_dir;
  },

  async loadDownloads() {
    const res = await fetch('/api/admin/models/downloads');
    const d = await res.json();
    this.downloadTasks = d.tasks || [];
    const active = this.downloadTasks.some(t => t.status === 'downloading' || t.status === 'pending');
    if (active && !this.downloadPollTimer) {
      this.downloadPollTimer = setInterval(() => this.pollDownloads(), 1500);
    } else if (!active && this.downloadPollTimer) {
      clearInterval(this.downloadPollTimer);
      this.downloadPollTimer = null;
    }
  },

  async pollDownloads() {
    await this.loadDownloads();
    const done = this.downloadTasks.some(t => t.status === 'completed');
    if (done) await this.loadInstalled();
  },

  filteredLocalCatalog() {
    return this.localCatalog.filter(m => {
      if (this.localRoleFilter && m.role !== this.localRoleFilter) return false;
      if (this.localFilter) {
        const q = this.localFilter.toLowerCase();
        return m.name.toLowerCase().includes(q) || m.description.toLowerCase().includes(q) ||
               m.tags.some(t => t.includes(q));
      }
      return true;
    });
  },

  filteredCloudCatalog() {
    return this.cloudCatalog.filter(p => {
      if (!this.cloudRegionFilter) return true;
      return p.region === this.cloudRegionFilter || p.region === 'both';
    });
  },

  selectLocalModel(m) {
    this.selectedLocalModel = m;
    const rec = m.variants.find(v => v.recommended) || m.variants[0];
    this.selectedQuant = rec?.quant || '';
  },

  selectProvider(p) {
    this.selectedProvider = p;
    const rec = p.models.find(m => m.recommended) || p.models[0];
    this.cloudSetup = { api_key: '', model_id: rec?.id || '', set_default: true };
  },

  async downloadModel() {
    if (!this.modelsDir) {
      this.toast('请先配置并保存模型存放目录', 'error');
      this.modelSubTab = 'paths';
      return;
    }
    if (!this.selectedLocalModel) return;
    try {
      const res = await fetch('/api/admin/models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: this.selectedLocalModel.id,
          quant: this.selectedQuant,
          models_dir: this.modelsDir,
        }),
      });
      const d = await res.json();
      if (!res.ok) throw new Error(d.detail || '下载失败');
      this.toast(d.task.status === 'completed' ? '模型已存在，无需下载' : '开始下载…', 'success');
      await this.loadDownloads();
    } catch (e) {
      this.toast(e.message, 'error');
    }
  },

  async registerInstalled(model) {
  },

  async activateLocalModel(model, catalogId) {
    if (!catalogId) {
      this.toast('请从模型库选择对应模型后激活', 'info');
      return;
    }
    try {
      const res = await fetch('/api/admin/models/register-local', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: catalogId,
          file_path: model.path,
          set_default: true,
        }),
      });
      const d = await res.json();
      if (!res.ok) throw new Error(d.detail || '激活失败');
      this.toast('本地模型已写入 config.yaml 并注册', 'success');
      await this.loadModels();
    } catch (e) {
      this.toast(e.message, 'error');
    }
  },

  async activateDownloaded(task) {
    if (!this.selectedLocalModel) {
      this.toast('请先选择对应的模型库条目', 'info');
      return;
    }
    await this.activateLocalModel({ path: task.dest_path }, this.selectedLocalModel.id);
  },

  async setupCloudProvider() {
    if (!this.selectedProvider || !this.cloudSetup.api_key || !this.cloudSetup.model_id) {
      this.toast('请填写 API Key 并选择模型', 'error');
      return;
    }
    try {
      const res = await fetch('/api/admin/models/setup-cloud', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider_id: this.selectedProvider.id,
          api_key: this.cloudSetup.api_key,
          model_id: this.cloudSetup.model_id,
          set_default: this.cloudSetup.set_default,
        }),
      });
      const d = await res.json();
      if (!res.ok) throw new Error(d.detail || '配置失败');
      this.toast(`${this.selectedProvider.name} 已写入 config.yaml`, 'success');
      this.cloudSetup.api_key = '';
      await this.loadModels();
      await this.loadAppConfig();
    } catch (e) {
      this.toast(e.message, 'error');
    }
  },

  regionLabel(r) {
    return { intl: '国际', cn: '国内', both: '国内外' }[r] || r;
  },

  roleLabel(r) {
    return { coding: '编程', general: '通用', fast: '快速', thinking: '深度' }[r] || r;
  },
};
