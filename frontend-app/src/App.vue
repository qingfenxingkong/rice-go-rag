<template>
  <div class="app-container">
    <header class="app-header">
      <div class="header-content">
        <h1>🌾 水稻 GO 知识图谱 RAG 问答系统</h1>
        <p class="subtitle">基于 Neo4j 知识图谱 + 向量检索 + DeepSeek LLM</p>
      </div>
      <div class="status-badge" :class="systemStatus.status">
        <span class="dot"></span>
        {{ systemStatus.text }}
      </div>
    </header>

    <main class="app-main">
      <div class="container">
        <div class="left-panel">
          <!-- 问题输入区 -->
          <section class="input-section">
            <h2>提出问题</h2>
            <textarea
              v-model="question"
              placeholder="例如：水稻线粒体相关的遗传过程有哪些？"
              class="question-input"
              @keydown.ctrl.enter="askQuestion"
            ></textarea>
            
            <div class="quick-examples">
              <button
                v-for="example in examples"
                :key="example"
                @click="question = example"
                class="example-btn"
              >
                {{ example }}
              </button>
            </div>

            <div class="ner-choice-box">
              <div class="ner-choice-top">
                <label class="switch-row">
                  <input type="checkbox" v-model="useNerAssist" />
                  <span>启用 NER 辅助检索</span>
                </label>
                <select v-model="nerMethodChoice" class="method-select" :disabled="!useNerAssist">
                  <option value="dict">dict</option>
                  <option value="vector">vector</option>
                  <option value="llm">llm</option>
                  <option value="ensemble">ensemble</option>
                </select>
              </div>
              <div class="method-grid">
                <div class="method-card" :class="{active: nerMethodChoice==='dict'}">
                  <h4>dict</h4>
                  <p><strong>优点：</strong>速度最快、可解释性强、稳定。</p>
                  <p><strong>缺点：</strong>对非标准表达和隐含语义覆盖弱。</p>
                </div>
                <div class="method-card" :class="{active: nerMethodChoice==='vector'}">
                  <h4>vector</h4>
                  <p><strong>优点：</strong>语义召回好，适合中英混合与近义表达。</p>
                  <p><strong>缺点：</strong>可能引入语义相近但不精确的结果。</p>
                </div>
                <div class="method-card" :class="{active: nerMethodChoice==='llm'}">
                  <h4>llm</h4>
                  <p><strong>优点：</strong>复杂语境理解强，能识别隐含实体。</p>
                  <p><strong>缺点：</strong>耗时高、成本高、结果波动更大。</p>
                </div>
                <div class="method-card" :class="{active: nerMethodChoice==='ensemble'}">
                  <h4>ensemble</h4>
                  <p><strong>优点：</strong>综合效果最好，精度与召回更平衡。</p>
                  <p><strong>缺点：</strong>流程更复杂，延迟高于单一方法。</p>
                </div>
              </div>
            </div>

            <button
              @click="askQuestion"
              :disabled="loading || !question.trim()"
              class="ask-btn"
            >
              <span v-if="!loading">🚀 发送问题</span>
              <span v-else>⏳ 生成中...</span>
            </button>
          </section>

          <!-- 答案展示区 -->
          <section class="answer-section" v-if="answer">
            <h2>AI 回答</h2>
            <div class="answer-content">{{ answer }}</div>
          </section>

          <!-- 错误提示 -->
          <section class="error-section" v-if="error">
            <h2>⚠️ 错误</h2>
            <div class="error-content">{{ error }}</div>
          </section>
        </div>

        <div class="right-panel">
          <!-- 知识图谱 -->
          <section class="graph-section">
            <h2>知识关系图谱</h2>
            <div id="graphContainer" class="graph-container"></div>
          </section>

          <!-- 相关术语 -->
          <section class="sources-section" v-if="sources.length > 0">
            <h2>相关 GO 术语 ({{ sources.length }})</h2>
            <div class="sources-list">
              <div v-for="source in sources" :key="source.go_id" class="source-item">
                <div class="source-header">
                  <span class="go-id">{{ source.go_id }}</span>
                  <span class="score">相似度: {{ (source.score * 100).toFixed(1) }}%</span>
                </div>
                <div class="source-name">{{ source.name }}</div>
                <div class="source-namespace">{{ source.namespace }}</div>
                <div class="source-desc">{{ source.description }}</div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { Network } from 'vis-network'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const question = ref('')
const answer = ref('')
const error = ref('')
const loading = ref(false)
const sources = ref([])
const useNerAssist = ref(false)
const nerMethodChoice = ref('ensemble')
const systemStatus = ref({ status: 'loading', text: '连接中...' })
let network = null

const examples = [
  '水稻线粒体相关的遗传过程有哪些？',
  '与水稻抗病性相关的生物过程 GO 术语有哪些？',
  '水稻光合作用相关的分子功能 GO 术语有哪些？'
]

// 初始化图谱
function initGraph() {
  const container = document.getElementById('graphContainer')
  if (!container) return

  const options = {
    physics: {
      enabled: true,
      stabilization: { iterations: 200 },
      barnesHut: {
        gravitationalConstant: -26000,
        centralGravity: 0.3,
        springLength: 200,
        springConstant: 0.04
      }
    },
    nodes: {
      shape: 'box',
      margin: 10,
      widthConstraint: { maximum: 200 },
      font: { size: 12, color: '#e5e7eb' },
      color: {
        background: '#1e293b',
        border: '#22c55e',
        highlight: { background: '#22c55e', border: '#16a34a' }
      }
    },
    edges: {
      color: { color: '#64748b', highlight: '#22c55e' },
      font: { size: 10, color: '#94a3b8' },
      smooth: { type: 'continuous' }
    }
  }

  if (network) network.destroy()
  network = new Network(container, { nodes: [], edges: [] }, options)
}

// 更新图谱
function updateGraph(nodes, edges) {
  if (!network) initGraph()
  
  const visNodes = nodes.map(n => ({
    id: n.id,
    label: n.label,
    title: `${n.label} (${n.namespace})`,
    color: {
      background: '#1e293b',
      border: '#22c55e',
      highlight: { background: '#22c55e', border: '#16a34a' }
    },
    font: { size: 12, color: '#e5e7eb' }
  }))

  const visEdges = edges.map(e => ({
    from: e.source,
    to: e.target,
    label: e.relationship,
    title: e.relationship,
    color: { color: '#64748b', highlight: '#22c55e' },
    font: { size: 10, color: '#94a3b8' }
  }))

  network.setData({ nodes: visNodes, edges: visEdges })
}

// 检查系统状态
async function checkHealth() {
  try {
    const resp = await axios.get(`${API_BASE}/health`)
    systemStatus.value = { status: 'ok', text: '✓ 系统正常' }
  } catch (err) {
    systemStatus.value = { status: 'error', text: '✗ 后端离线' }
  }
}

// 提问
async function askQuestion() {
  if (!question.value.trim()) return

  loading.value = true
  error.value = ''
  answer.value = ''
  sources.value = []

  try {
    // 获取流式答案
    const response = await axios.post(`${API_BASE}/ask_stream`, {
      question: question.value,
      use_ner: useNerAssist.value,
      ner_method: nerMethodChoice.value,
      ner_ensemble_mode: 'balanced',
    }, {
      responseType: 'stream'
    })

    answer.value = ''
    const reader = response.data.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      answer.value += decoder.decode(value, { stream: true })
    }

    // 获取图谱
    try {
      const graphResp = await axios.post(`${API_BASE}/graph`, {
        question: question.value
      })
      if (graphResp.data.nodes && graphResp.data.nodes.length > 0) {
        updateGraph(graphResp.data.nodes, graphResp.data.edges || [])
      }
    } catch (graphErr) {
      console.warn('获取图谱失败:', graphErr)
    }

    // 获取相关术语
    try {
      const askResp = await axios.post(`${API_BASE}/ask`, {
        question: question.value,
        use_ner: useNerAssist.value,
        ner_method: nerMethodChoice.value,
        ner_ensemble_mode: 'balanced',
      })
      sources.value = askResp.data.sources || []
    } catch (sourcesErr) {
      console.warn('获取相关术语失败:', sourcesErr)
    }
  } catch (err) {
    error.value = `请求失败: ${err.message}`
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  checkHealth()
  initGraph()
  setInterval(checkHealth, 30000)
})
</script>

<style scoped>
.app-container {
  min-height: 100vh;
  background: radial-gradient(circle at top, #1e293b 0, #020617 55%);
  display: flex;
  flex-direction: column;
}

.app-header {
  background: linear-gradient(135deg, #020617 0, #0b1120 100%);
  border-bottom: 1px solid rgba(148, 163, 184, 0.25);
  padding: 24px 32px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-content h1 {
  font-size: 28px;
  margin-bottom: 8px;
  background: linear-gradient(135deg, #22c55e, #16a34a);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.subtitle {
  font-size: 14px;
  color: #9ca3af;
}

.status-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
}

.status-badge.ok {
  background: rgba(34, 197, 94, 0.12);
  color: #22c55e;
  border: 1px solid rgba(34, 197, 94, 0.3);
}

.status-badge.error {
  background: rgba(249, 115, 115, 0.12);
  color: #f97373;
  border: 1px solid rgba(249, 115, 115, 0.3);
}

.status-badge.loading {
  background: rgba(59, 130, 246, 0.12);
  color: #3b82f6;
  border: 1px solid rgba(59, 130, 246, 0.3);
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
  display: inline-block;
}

.app-main {
  flex: 1;
  padding: 32px;
  overflow: auto;
}

.container {
  max-width: 1400px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.left-panel, .right-panel {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

section {
  background: linear-gradient(135deg, #020617 0, #020617 50%, #0b1120 100%);
  border-radius: 18px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  padding: 20px;
  box-shadow: 0 24px 80px rgba(15, 23, 42, 0.9);
}

h2 {
  font-size: 16px;
  margin-bottom: 16px;
  color: #e5e7eb;
}

.question-input {
  width: 100%;
  min-height: 100px;
  padding: 12px;
  border-radius: 12px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  background: #020617;
  color: #e5e7eb;
  font-size: 14px;
  line-height: 1.5;
  resize: vertical;
  font-family: inherit;
}

.question-input:focus {
  outline: none;
  border-color: #22c55e;
  box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.4);
}

.quick-examples {
  display: flex;
  gap: 8px;
  margin: 12px 0;
  flex-wrap: wrap;
}

.example-btn {
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.6);
  background: rgba(15, 23, 42, 0.9);
  color: #9ca3af;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s;
}

.example-btn:hover {
  border-color: #22c55e;
  color: #22c55e;
}

.ask-btn {
  padding: 12px 24px;
  border-radius: 999px;
  border: none;
  background: linear-gradient(135deg, #22c55e, #16a34a);
  color: #020617;
  font-weight: 600;
  font-size: 14px;
  cursor: pointer;
  box-shadow: 0 10px 30px rgba(34, 197, 94, 0.35);
  transition: all 0.2s;
}

.ask-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 15px 40px rgba(34, 197, 94, 0.45);
}

.ask-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.ner-choice-box {
  margin-bottom: 12px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  background: rgba(15, 23, 42, 0.55);
  border-radius: 12px;
  padding: 12px;
}

.ner-choice-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}

.switch-row {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #cbd5e1;
  font-size: 13px;
}

.method-select {
  background: #020617;
  border: 1px solid rgba(148, 163, 184, 0.3);
  color: #e5e7eb;
  border-radius: 8px;
  padding: 6px 10px;
}

.method-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.method-card {
  border: 1px solid rgba(100, 116, 139, 0.3);
  border-radius: 10px;
  background: rgba(2, 6, 23, 0.65);
  padding: 8px 10px;
}

.method-card.active {
  border-color: rgba(34, 197, 94, 0.7);
  box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.25);
}

.method-card h4 {
  font-size: 12px;
  color: #86efac;
  margin-bottom: 4px;
}

.method-card p {
  font-size: 11px;
  color: #cbd5e1;
  line-height: 1.45;
}

.answer-content {
  background: radial-gradient(circle at top left, #0f172a 0, #020617 55%);
  border: 1px solid rgba(51, 65, 85, 0.9);
  border-radius: 12px;
  padding: 12px;
  font-size: 14px;
  line-height: 1.7;
  max-height: 400px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
}

.error-content {
  background: rgba(249, 115, 115, 0.12);
  border: 1px solid rgba(249, 115, 115, 0.3);
  border-radius: 12px;
  padding: 12px;
  color: #fca5a5;
  font-size: 13px;
}

#graphContainer {
  width: 100%;
  height: 400px;
  border-radius: 12px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  background: #020617;
}

.sources-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 400px;
  overflow: auto;
}

.source-item {
  background: radial-gradient(circle at left, rgba(37, 99, 235, 0.3), transparent 60%);
  border: 1px solid rgba(30, 64, 175, 0.6);
  border-radius: 9px;
  padding: 10px;
  font-size: 12px;
}

.source-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 6px;
}

.go-id {
  font-weight: 600;
  color: #bfdbfe;
}

.score {
  color: #a5b4fc;
  font-size: 11px;
}

.source-name {
  font-weight: 500;
  color: #e5e7eb;
  margin-bottom: 4px;
}

.source-namespace {
  color: #bae6fd;
  font-size: 11px;
  margin-bottom: 4px;
}

.source-desc {
  color: #d1d5db;
  line-height: 1.5;
}

@media (max-width: 1024px) {
  .container {
    grid-template-columns: 1fr;
  }
  
  .app-header {
    flex-direction: column;
    gap: 16px;
    align-items: flex-start;
  }
}

@media (max-width: 640px) {
  .app-main {
    padding: 16px;
  }
  
  .app-header {
    padding: 16px;
  }
  
  h1 {
    font-size: 20px;
  }

  .method-grid {
    grid-template-columns: 1fr;
  }
  
  #graphContainer {
    height: 250px;
  }
}
</style>
