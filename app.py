"""
Motor de Orquestração de Stablecoins — Demo PoC (WhatsApp + Dashboard B2B)
==========================================================================
Arquivo único Flask que integra:
  • Webhook Twilio (WhatsApp Sandbox) — rota POST /whatsapp
  • Dashboard B2B em tempo real       — rota GET  /
  • API de polling                    — rota GET  /api/status
  • Simulação on-chain                — rota POST /simulate

Variáveis de ambiente necessárias:
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER
"""

import os
import re
import time
import uuid
import threading
from datetime import datetime

from flask import Flask, request, jsonify, render_template_string
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client

# ──────────────────────────────────────────────
# App & Config
# ──────────────────────────────────────────────
app = Flask(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "")  # ex: whatsapp:+14155238886

STARTUP_NAME = "CryptoFlow"
FAKE_POLYGON_ADDR = "0x7a3F...e91B"
CAMBIO_BRL = 5.00   # câmbio simulado
SPREAD_PCT = 0.01   # 1 %

# Estado global em memória (suficiente para o demo)
PENDING_TXS: dict = {}          # tx_id -> {...}
SIMULATION_LOGS: dict = {}       # tx_id -> [linhas de log]
SIMULATION_DONE: dict = {}       # tx_id -> resultado final

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _mask_phone(number: str) -> str:
    """Censura parcialmente o telefone: +55 11 9****-1234"""
    digits = re.sub(r"\D", "", number)
    if len(digits) >= 10:
        return f"+{digits[:2]} {digits[2:4]} {digits[4]}****-{digits[-4:]}"
    return number


def _extract_amount(text: str) -> float | None:
    """Extrai o primeiro número (int ou float) da mensagem."""
    # tenta pegar valores tipo 2000, 2.500, 2000.50 etc.
    patterns = [
        r"[\d]+[.,]?\d*",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            raw = m.group().replace(",", ".")
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def _send_whatsapp(to: str, body: str):
    """Envia mensagem push via Twilio REST API."""
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    from_number = TWILIO_WHATSAPP_NUMBER
    if not from_number.startswith("whatsapp:"):
        from_number = f"whatsapp:{from_number}"
    if not to.startswith("whatsapp:"):
        to = f"whatsapp:{to}"
    client.messages.create(body=body, from_=from_number, to=to)


# ──────────────────────────────────────────────
# PARTE 1 — Webhook WhatsApp
# ──────────────────────────────────────────────

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get("Body", "").strip()
    sender = request.values.get("From", "")  # whatsapp:+5511999991234

    amount = _extract_amount(incoming_msg)
    if amount is None:
        amount = 1000.0  # fallback para o demo

    tx_id = str(uuid.uuid4())[:8]
    PENDING_TXS[tx_id] = {
        "sender": sender,
        "amount_usd": amount,
        "status": "Aguardando USDC",
        "created_at": datetime.utcnow().isoformat(),
        "masked_phone": _mask_phone(sender),
    }

    resp = MessagingResponse()
    resp.message(
        f"🤖 Olá! Sou a IA financeira da *{STARTUP_NAME}*.\n\n"
        f"Criei seu endereço exclusivo na rede Polygon:\n"
        f"`{FAKE_POLYGON_ADDR}`\n\n"
        f"Envie este endereço para a sua empresa nos EUA.\n"
        f"Nosso sistema está monitorando a blockchain em tempo real... ⏳"
    )
    return str(resp), 200, {"Content-Type": "application/xml"}


# ──────────────────────────────────────────────
# PARTE 2 — Dashboard B2B + API
# ──────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    txs = []
    for tx_id, data in PENDING_TXS.items():
        txs.append({
            "tx_id": tx_id,
            "masked_phone": data["masked_phone"],
            "amount_usd": data["amount_usd"],
            "status": data["status"],
            "created_at": data["created_at"],
        })
    return jsonify({"transactions": txs})


@app.route("/api/simulation/<tx_id>")
def api_simulation_status(tx_id):
    logs = SIMULATION_LOGS.get(tx_id, [])
    done = SIMULATION_DONE.get(tx_id)
    return jsonify({"logs": logs, "result": done})


@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.get_json(force=True)
    tx_id = data.get("tx_id")

    if tx_id not in PENDING_TXS:
        return jsonify({"error": "Transação não encontrada"}), 404

    tx = PENDING_TXS[tx_id]
    amount_usd = tx["amount_usd"]
    sender = tx["sender"]

    # Limpa estado anterior
    SIMULATION_LOGS[tx_id] = []
    SIMULATION_DONE[tx_id] = None

    def _run_simulation():
        logs = SIMULATION_LOGS[tx_id]

        steps = [
            ("[BLOCKCHAIN] Webhook recebido: USDC detectados na rede Polygon.", 1.5),
            ("[COMPLIANCE] Verificando CPF e Sanções OFAC... Status: APROVADO ✅", 1.8),
            (f"[MOTOR FX] Cotando USD/BRL no Banco Parceiro... Câmbio travado a R$ {CAMBIO_BRL:.2f}.", 1.5),
            (f"[UNIT ECONOMICS] Retendo spread da plataforma ({SPREAD_PCT*100:.0f}%).", 1.2),
            ("[PIX API] Preparando liquidação Pix instantânea...", 1.5),
            ("[PIX API] Pix enviado com sucesso! ✅", 1.0),
        ]

        for msg, delay in steps:
            time.sleep(delay)
            logs.append(msg)

        bruto_brl = amount_usd * CAMBIO_BRL
        lucro = bruto_brl * SPREAD_PCT
        liquido = bruto_brl - lucro

        SIMULATION_DONE[tx_id] = {
            "amount_usd": amount_usd,
            "bruto_brl": round(bruto_brl, 2),
            "lucro_startup": round(lucro, 2),
            "liquido_pix": round(liquido, 2),
            "cambio": CAMBIO_BRL,
        }

        # Grand Finale — mensagem push para o WhatsApp do investidor
        try:
            _send_whatsapp(
                sender,
                f"✅ *BIP! Dinheiro na conta!*\n\n"
                f"A blockchain confirmou os fundos. Seus dólares foram "
                f"convertidos instantaneamente e o valor líquido de "
                f"*R$ {liquido:,.2f}* acabou de cair na sua conta via Pix.\n\n"
                f"Você economizou 5 dias de espera do SWIFT e altas taxas "
                f"bancárias! 🚀",
            )
        except Exception as e:
            logs.append(f"[ALERTA] Erro ao enviar WhatsApp: {e}")

        # Remove transação do pendente
        PENDING_TXS.pop(tx_id, None)

    threading.Thread(target=_run_simulation, daemon=True).start()
    return jsonify({"ok": True, "tx_id": tx_id})


# ──────────────────────────────────────────────
# Dashboard HTML (embutido)
# ──────────────────────────────────────────────

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{{ startup }} — Motor de Orquestração</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    body { font-family: 'Inter', sans-serif; }
    .glow { box-shadow: 0 0 30px rgba(99, 102, 241, 0.3); }
    .terminal {
      background: #0f172a;
      color: #22d3ee;
      font-family: 'Courier New', monospace;
      font-size: 0.85rem;
      line-height: 1.8;
      padding: 1.5rem;
      border-radius: 0.75rem;
      min-height: 200px;
      overflow-y: auto;
    }
    .terminal .line { opacity: 0; animation: fadeIn 0.4s forwards; }
    @keyframes fadeIn { to { opacity: 1; } }
    @keyframes pulse-border {
      0%, 100% { border-color: rgba(99, 102, 241, 0.4); }
      50% { border-color: rgba(99, 102, 241, 1); }
    }
    .pulse-border { animation: pulse-border 2s ease-in-out infinite; }
    .card-appear { animation: slideUp 0.5s ease-out forwards; }
    @keyframes slideUp {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body class="bg-gray-950 text-white min-h-screen">

  <!-- Header -->
  <header class="border-b border-gray-800 bg-gray-950/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center text-lg font-bold">C</div>
        <div>
          <h1 class="text-xl font-bold tracking-tight">{{ startup }}</h1>
          <p class="text-xs text-gray-500">Motor de Orquestração de Stablecoins</p>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <span id="status-dot" class="w-2.5 h-2.5 rounded-full bg-green-500 animate-pulse"></span>
        <span class="text-sm text-gray-400">Sistema Online</span>
      </div>
    </div>
  </header>

  <main class="max-w-6xl mx-auto px-6 py-10">

    <!-- Empty State -->
    <div id="empty-state" class="text-center py-20">
      <div class="text-6xl mb-6">📡</div>
      <h2 class="text-2xl font-bold text-gray-300 mb-2">Monitorando Blockchain...</h2>
      <p class="text-gray-500 max-w-md mx-auto">
        Envie uma mensagem pelo WhatsApp para iniciar uma transferência.
        O sistema detectará automaticamente a intenção de recebimento.
      </p>
      <div class="mt-6 inline-flex items-center gap-2 text-sm text-indigo-400 bg-indigo-950/50 px-4 py-2 rounded-full">
        <span class="w-2 h-2 rounded-full bg-indigo-400 animate-pulse"></span>
        Aguardando mensagens...
      </div>
    </div>

    <!-- Transaction Card -->
    <div id="tx-card" class="hidden card-appear">
      <div class="border border-indigo-500/30 rounded-2xl p-8 bg-gray-900/60 glow">
        <div class="flex items-start justify-between mb-6">
          <div>
            <span class="text-xs font-medium text-indigo-400 bg-indigo-950 px-3 py-1 rounded-full">
              TRANSAÇÃO DETECTADA
            </span>
            <h2 class="text-2xl font-bold mt-3" id="tx-phone">—</h2>
            <p class="text-gray-500 text-sm mt-1">via WhatsApp</p>
          </div>
          <div class="text-right">
            <p class="text-sm text-gray-500">Valor declarado</p>
            <p class="text-3xl font-extrabold text-green-400" id="tx-amount">—</p>
          </div>
        </div>

        <div class="flex items-center gap-3 mb-8">
          <span class="text-xs text-yellow-400 bg-yellow-950 px-3 py-1 rounded-full flex items-center gap-1.5">
            <span class="w-1.5 h-1.5 rounded-full bg-yellow-400 animate-pulse"></span>
            <span id="tx-status">Aguardando USDC</span>
          </span>
          <span class="text-xs text-gray-600" id="tx-time">—</span>
        </div>

        <!-- Simulate Button -->
        <button
          id="btn-simulate"
          onclick="startSimulation()"
          class="w-full py-5 rounded-xl bg-indigo-600 hover:bg-indigo-500 transition-all
                 text-lg font-bold tracking-wide flex items-center justify-center gap-3
                 hover:scale-[1.01] active:scale-[0.99]"
        >
          ⚡ SIMULAR CHEGADA DE USDC ON-CHAIN
        </button>
      </div>

      <!-- Terminal -->
      <div id="terminal-wrapper" class="hidden mt-8">
        <h3 class="text-sm font-semibold text-gray-400 mb-3 flex items-center gap-2">
          <span class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
          PROCESSAMENTO EM TEMPO REAL
        </h3>
        <div id="terminal" class="terminal"></div>
      </div>

      <!-- Result Cards -->
      <div id="result-cards" class="hidden mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="card-appear bg-gray-900 border border-gray-800 rounded-xl p-6 text-center">
          <p class="text-xs text-gray-500 uppercase tracking-wider mb-1">Valor Bruto</p>
          <p class="text-3xl font-extrabold text-white" id="res-bruto">—</p>
        </div>
        <div class="card-appear bg-gray-900 border border-emerald-800 rounded-xl p-6 text-center" style="animation-delay:0.15s">
          <p class="text-xs text-emerald-400 uppercase tracking-wider mb-1">Lucro {{ startup }}</p>
          <p class="text-3xl font-extrabold text-emerald-400" id="res-lucro">—</p>
        </div>
        <div class="card-appear bg-gray-900 border border-gray-800 rounded-xl p-6 text-center" style="animation-delay:0.3s">
          <p class="text-xs text-gray-500 uppercase tracking-wider mb-1">Líquido via Pix</p>
          <p class="text-3xl font-extrabold text-white" id="res-liquido">—</p>
        </div>
      </div>

      <!-- WhatsApp Sent Banner -->
      <div id="wpp-banner" class="hidden mt-6 card-appear">
        <div class="bg-green-950/60 border border-green-700/40 rounded-xl p-5 flex items-center gap-4">
          <div class="text-3xl">📲</div>
          <div>
            <p class="font-bold text-green-400">Mensagem enviada ao WhatsApp do cliente!</p>
            <p class="text-sm text-green-300/70">O usuário recebeu a confirmação de Pix no celular dele.</p>
          </div>
        </div>
      </div>
    </div>
  </main>

  <!-- Footer -->
  <footer class="border-t border-gray-800 mt-20 py-6 text-center text-xs text-gray-600">
    {{ startup }} © 2026 — Motor de Orquestração de Stablecoins — Demo PoC
  </footer>

  <script>
    let currentTxId = null;
    let pollSimInterval = null;
    let lastLogCount = 0;

    // Poll for new transactions
    async function pollStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();

        if (data.transactions.length > 0 && !currentTxId) {
          const tx = data.transactions[0];
          currentTxId = tx.tx_id;
          showTransaction(tx);
        } else if (data.transactions.length === 0 && !pollSimInterval) {
          currentTxId = null;
          // keep result on screen if simulation just ended
        }
      } catch (e) {
        console.error('Poll error:', e);
      }
    }

    function showTransaction(tx) {
      document.getElementById('empty-state').classList.add('hidden');
      document.getElementById('tx-card').classList.remove('hidden');
      document.getElementById('tx-phone').textContent = tx.masked_phone;
      document.getElementById('tx-amount').textContent = `US$ ${Number(tx.amount_usd).toLocaleString('pt-BR', {minimumFractionDigits: 2})}`;
      document.getElementById('tx-status').textContent = tx.status;
      document.getElementById('tx-time').textContent = new Date(tx.created_at + 'Z').toLocaleTimeString('pt-BR');

      // Reset UI
      document.getElementById('btn-simulate').classList.remove('hidden');
      document.getElementById('btn-simulate').disabled = false;
      document.getElementById('terminal-wrapper').classList.add('hidden');
      document.getElementById('terminal').innerHTML = '';
      document.getElementById('result-cards').classList.add('hidden');
      document.getElementById('wpp-banner').classList.add('hidden');
    }

    async function startSimulation() {
      if (!currentTxId) return;

      const btn = document.getElementById('btn-simulate');
      btn.disabled = true;
      btn.innerHTML = '<span class="animate-spin inline-block w-5 h-5 border-2 border-white border-t-transparent rounded-full"></span> Processando...';

      document.getElementById('terminal-wrapper').classList.remove('hidden');
      document.getElementById('terminal').innerHTML = '';
      lastLogCount = 0;

      await fetch('/simulate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({tx_id: currentTxId}),
      });

      pollSimInterval = setInterval(() => pollSimulation(currentTxId), 800);
    }

    async function pollSimulation(txId) {
      try {
        const res = await fetch(`/api/simulation/${txId}`);
        const data = await res.json();

        const terminal = document.getElementById('terminal');
        for (let i = lastLogCount; i < data.logs.length; i++) {
          const div = document.createElement('div');
          div.className = 'line';
          div.textContent = '> ' + data.logs[i];
          terminal.appendChild(div);
          terminal.scrollTop = terminal.scrollHeight;
        }
        lastLogCount = data.logs.length;

        if (data.result) {
          clearInterval(pollSimInterval);
          pollSimInterval = null;
          showResult(data.result);
        }
      } catch (e) {
        console.error('Sim poll error:', e);
      }
    }

    function fmt(v) {
      return 'R$ ' + Number(v).toLocaleString('pt-BR', {minimumFractionDigits: 2});
    }

    function showResult(r) {
      document.getElementById('btn-simulate').classList.add('hidden');

      document.getElementById('result-cards').classList.remove('hidden');
      document.getElementById('res-bruto').textContent = fmt(r.bruto_brl);
      document.getElementById('res-lucro').textContent = fmt(r.lucro_startup);
      document.getElementById('res-liquido').textContent = fmt(r.liquido_pix);

      setTimeout(() => {
        document.getElementById('wpp-banner').classList.remove('hidden');
      }, 600);
    }

    setInterval(pollStatus, 2000);
    pollStatus();
  </script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML, startup=STARTUP_NAME)


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🚀 {STARTUP_NAME} — Motor de Orquestração de Stablecoins")
    print(f"   Dashboard:  http://localhost:5000")
    print(f"   Webhook:    http://localhost:5000/whatsapp\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
