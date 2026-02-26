# CryptoFlow — Guia de Setup do Demo (WhatsApp + Dashboard B2B)

## 1. Pré-requisitos

- Python 3.10+
- Uma conta Twilio (gratuita funciona)
- ngrok instalado ([ngrok.com](https://ngrok.com))
- Um celular com WhatsApp

---

## 2. Instalar dependências

```bash
pip install -r requirements-demo.txt
```

Ou manualmente:

```bash
pip install flask twilio
```

---

## 3. Configurar o Twilio WhatsApp Sandbox

### 3.1 Criar conta no Twilio

1. Acesse [twilio.com/try-twilio](https://www.twilio.com/try-twilio) e crie uma conta gratuita.
2. Confirme seu e-mail e número de telefone.

### 3.2 Ativar o WhatsApp Sandbox

1. No painel do Twilio, vá em **Messaging > Try it out > Send a WhatsApp message**.
2. Você verá uma tela com instruções tipo:
   > Envie a mensagem `join <palavra-secreta>` para o número **+1 415 523 8886** pelo WhatsApp.
3. Abra o WhatsApp no celular do investidor e envie essa mensagem. Pronto — o celular está conectado ao Sandbox.

### 3.3 Pegar as credenciais

1. Vá em **Account > API keys & tokens** (ou na Dashboard inicial).
2. Copie o **Account SID** e o **Auth Token**.
3. O número do WhatsApp Sandbox é geralmente `+14155238886`.

### 3.4 Configurar variáveis de ambiente

```bash
export TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export TWILIO_AUTH_TOKEN="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export TWILIO_WHATSAPP_NUMBER="whatsapp:+14155238886"
```

> No Windows use `set` ao invés de `export`.

---

## 4. Rodar o servidor Flask

```bash
python app.py
```

Você verá:

```
🚀 CryptoFlow — Motor de Orquestração de Stablecoins
   Dashboard:  http://localhost:5000
   Webhook:    http://localhost:5000/whatsapp
```

---

## 5. Expor com ngrok

Em **outro terminal**, rode:

```bash
ngrok http 5000
```

O ngrok vai gerar uma URL pública tipo:

```
https://a1b2c3d4.ngrok-free.app
```

### 5.1 Configurar o Webhook no Twilio

1. Volte ao painel do Twilio em **Messaging > Try it out > Send a WhatsApp message**.
2. Na seção **Sandbox Configuration**, cole a URL do ngrok com `/whatsapp` no final:
   ```
   https://a1b2c3d4.ngrok-free.app/whatsapp
   ```
3. Selecione o método **POST**.
4. Clique em **Save**.

---

## 6. Testar o Demo

### Fluxo completo:

1. **Investidor (celular):** Abre o WhatsApp e envia para o número do Sandbox:
   > "Vou receber 2000 dólares"

2. **Investidor (celular):** Recebe resposta automática da IA com endereço Polygon.

3. **Você (notebook/projetor):** Abre `http://localhost:5000` no navegador. O card da transação aparece automaticamente.

4. **Você (notebook/projetor):** Clica no botão **"⚡ SIMULAR CHEGADA DE USDC ON-CHAIN"**.

5. **Plateia observa:** Terminal animado mostrando cada etapa do processamento (blockchain, compliance, FX, Pix).

6. **Investidor (celular):** Recebe mensagem WhatsApp confirmando o Pix! Momento "wow".

---

## 7. Dicas para a apresentação

- **Antes de entrar na sala:** Teste o fluxo completo sozinho para garantir que o Sandbox está ativo e o ngrok funcionando.
- **Wi-Fi:** Certifique-se de que tanto o notebook quanto o celular do investidor estejam com internet estável.
- **Sandbox expira:** O opt-in do Sandbox expira após ~72h. Peça ao investidor para enviar a mensagem `join <palavra>` ali na hora — isso leva 5 segundos e pode ser parte da narrativa ("vamos conectar seu WhatsApp ao nosso sistema").
- **Valores:** O sistema extrai automaticamente números da mensagem. Se o investidor digitar algo sem número, o fallback é US$ 1.000.

---

## Arquitetura resumida

```
┌──────────────┐     WhatsApp      ┌────────────┐     Webhook     ┌─────────────┐
│  Celular do  │ ───────────────▶  │   Twilio   │ ──────────────▶ │  Flask API  │
│  Investidor  │ ◀───────────────  │  Sandbox   │ ◀────────────── │  (app.py)   │
└──────────────┘   Msg Push (API)  └────────────┘    TwiML Reply  └──────┬──────┘
                                                                         │
                                                              ┌──────────▼──────────┐
                                                              │  Dashboard B2B      │
                                                              │  (localhost:5000)   │
                                                              │  Tela do Projetor   │
                                                              └─────────────────────┘
```
