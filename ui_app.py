# Interface do usuário em Streamlit

import os
import json
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import requests
import logging
import time
import base64
import re
from collections import defaultdict

# ======================================================
#                Configuração inicial
# ======================================================
st.set_page_config(page_title="UBChat - Unimed", page_icon="💬", layout="centered")

# ---- Variáveis de ambiente ---- #
_ = load_dotenv(override=True)

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
API_URL = os.getenv("API_URL")
POD_ID = os.getenv("POD_ID", "")
BACKEND_PORT = os.getenv("BACKEND_PORT", "8000")

if POD_ID:
    BACKEND_URL = f"https://{POD_ID}-{BACKEND_PORT}.proxy.runpod.net"
else:
    BACKEND_URL = os.getenv("BACKEND_URL", f"http://localhost:{BACKEND_PORT}")

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

UNAVAILABLE_MSG = (
    "🚧 **Servidor indisponível no momento.** "
    "Não foi possível processar sua pergunta. "
    "Tente novamente em alguns minutos."
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================================================
#                Funções Helper
# ======================================================

def now_str() -> str:
    """Timestamp no fuso America/Sao_Paulo."""
    return datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y-%m-%d %H:%M:%S")

def autenticar_usuario(usuario, senha):
    """Envia os dados via GET com a senha em base64"""

    if not API_URL:
        st.error("API_URL não configurada.")
        return None

    senha_b64 = base64.b64encode(senha.encode('utf-8')).decode('utf-8')

    # Parâmetros na URL
    params = {
        "usuario": usuario,
        "senha": senha_b64,
        "ip": "",
        "origem": "assistente_chat"
    }

    # Cabeçalhos da requisição
    headers = {
        "Authorization": AUTH_TOKEN,
        "Accept": "application/json"
    }

    try:
        resposta = requests.get(API_URL, params=params, headers=headers, timeout=15)
        if resposta.status_code == 200:
            return resposta.json()
        return None
    except Exception as e:
        logger.error(f"Erro ao autenticar usuário: {e}")
        st.error("Erro ao conectar com a API")
        return None

def get_or_create_session_id() -> str:
    """
    Gera ou recupera um session_id
    """
    if "sid" not in st.session_state:
        qs = st.query_params
        sid = qs.get("sid", None)

        if not sid:
            sid = str(uuid.uuid4())
            st.query_params.update({"sid": sid})
        st.session_state["sid"] = sid
    return st.session_state["sid"]

def check_backend(timeout: int = 10) -> dict:
    """
    Verifica se o backend está saudável via rota /health.
    Retorna um dicionário com status e detalhes.
    """
    url = f"{BACKEND_URL.rstrip('/')}/health"

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        status = data.get("status", "UNKNOWN")
        return {
            "status": status.upper(),
            "ok": status.upper() == "OK",
            "raw": data
        }
    except requests.RequestException as e:
        logger.error(f"Erro ao verificar backend: {e}")
        return {
            "status": "ERROR",
            "ok": False,
            "raw": {"error": str(e)}
        }

def call_backend(message: str, session_id: str, force_rag: bool = True, timeout: int = 480) -> dict:
    """
    Chama o endpoint /chat do backend.

    Args:
        message: Pergunta do usuário
        session_id: ID da sessão
        force_rag: Se True, força o uso de RAG
        timeout: Timeout da requisição

    Returns:
        Dicionário com resposta e metadados
    """
    url = f"{BACKEND_URL.rstrip('/')}/chat"

    # Recupera histórico de chat da sessão
    chat_history = st.session_state.get("chat_history", [])

    headers = {
        "Content-Type": "application/json",
        "X-Session-Id": session_id
    }

    payload = {
        "question": message,
        "chat_history": chat_history
    }

    try:
        start_time = time.time()
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        latency_ms = (time.time() - start_time) * 1000

        obj = r.json()

        # Extrai informações da resposta
        answer = obj.get("answer", "")
        route = obj.get("route", "")
        confidence = obj.get("confidence", 0.0)
        reasoning = obj.get("reasoning", "")
        documents = obj.get("documents", [])
        clarifying_questions = obj.get("clarifying_questions", [])

        return {
            "reply": answer,
            "sources": documents,
            "route": route,
            "confidence": confidence,
            "reasoning": reasoning,
            "clarifying_questions": clarifying_questions,
            "latency_ms": latency_ms,
            "raw": obj,
        }

    except requests.Timeout:
        logger.error("Timeout ao chamar backend")
        return {
            "reply": "⏱️ A requisição demorou muito tempo. Tente novamente com uma pergunta mais específica.",
            "sources": [],
            "route": "error",
            "raw": {"error": "timeout"},
        }
    except requests.RequestException as e:
        logger.error(f"Erro de requisição ao backend: {e}")
        return {
            "reply": f"❌ Erro ao chamar o backend: {str(e)}",
            "sources": [],
            "route": "error",
            "raw": {"error": str(e)},
        }
    except Exception as e:
        logger.error(f"Erro inesperado ao chamar backend: {e}")
        return {
            "reply": f"❌ Erro inesperado: {str(e)}",
            "sources": [],
            "route": "error",
            "raw": {"error": str(e)},
        }

def reinicia_conversa(session_id: str, rotate_sid: bool = True) -> None:
    """
    Limpa a UI e o histórico de chat.
    Se rotate_sid=True, gera um novo session_id após limpar.
    """
    # Limpa histórico de chat
    st.session_state.chat_history = []

    # Limpa estado da UI
    st.session_state.messages = []
    st.session_state.feedback_enviado = False
    st.session_state.show_feedback = False
    st.session_state.pop("last_question", None)
    st.session_state.pop("last_answer", None)

    # Opcional: Gira o SID (novo chat "frio")
    if rotate_sid:
        new_sid = str(uuid.uuid4())
        st.session_state['sid'] = new_sid
        st.query_params.update({"sid": new_sid})

    logger.info(f"Conversa reiniciada para sessão {session_id}")
    st.rerun()

def registrar_historico(
    usuario: str,
    pergunta: str,
    resposta: str,
    session_id: str,
    latencia: float,
    mode: str,
    generation_info: dict,
    usage_info: dict,
    extra: dict,
) -> None:
    """Registra histórico de interações em arquivo de log."""

    record = {
        "timestamp": now_str(),
        "usuario": usuario,
        "pergunta": pergunta,
        "resposta": resposta,
        "session_id": session_id,
        "latencia": latencia,
        "modo": mode,
        "generation_info": generation_info or {},
        "usage_info": usage_info or {},
        "extra": extra or {},
        "app_version": APP_VERSION,
    }

    log_path = Path("monitoramento/history.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def registrar_erro(usuario: str, session_id: str, pergunta: str, mensagem: str):
    """Registra erros em arquivo de log."""

    record = {
        "timestamp": now_str(),
        "usuario": usuario,
        "session_id": session_id,
        "pergunta": pergunta,
        "erro": mensagem
    }

    log_path = Path("monitoramento/erros.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def render_login():
    """Renderiza a tela de login."""
    login_ph = st.empty()
    with login_ph.container():
        st.markdown(
            """
            <div style="text-align:center; margin-bottom: 1rem;">
                <h2 style="margin:0;">Login</h2>
                <p style="opacity:0.75; margin-top:0.25rem;">Acesse para usar o assistente</p>
            </div>
            """, unsafe_allow_html=True,
        )

        user = st.text_input("Usuário:", key="login_user")
        password = st.text_input("Senha", type="password", key="login_pass")

        col1, col2 = st.columns(2)
        entrar = col1.button("Entrar", use_container_width=True)
        limpar = col2.button("Limpar", use_container_width=True)

        if entrar:
            if not user or not password:
                st.warning("Por favor, preencha usuário e senha.")
            else:
                autenticacao = autenticar_usuario(user, password)
                if autenticacao and autenticacao.get("Result") == 1:
                    st.session_state.logged_in = True
                    st.session_state.dados_usuario = autenticacao
                    st.session_state.usuario = user
                    # remove a UI de login imediatamente
                    login_ph.empty()
                    st.rerun()
                else:
                    st.error("Usuário ou senha inválidos")

        if limpar:
            for k in ("login_user", "login_pass"):
                st.session_state.pop(k, None)
            st.rerun()

def render_header():
    """Renderiza o cabeçalho da aplicação."""
    st.markdown(
        """
        <div style="max-width: 1000px; margin: 0 auto; text-align:center;">
            <h1 style="margin-bottom: 0.25rem; color: #00995D;">
                💬 <span style="color: #B1D34B;">UB</span> Chat
            </h1>
            <p style="margin-top: 0; font-size: 0.95rem; color: #5B5C65;">
                Faça perguntas e tire suas dúvidas com o assistente de conversação da Unimed.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _nome_fonte(meta: dict) -> str:
    """Extrai um nome 'amigável' da fonte a partir de vários campos possíveis."""
    candidatos = [
        meta.get("source"),
        meta.get("doc_id"),
        meta.get("file_path"),
        meta.get("file"),
        meta.get("document_id"),
    ]
    for c in candidatos:
        if not c:
            continue
        # Pega só o último trecho do caminho/URL e remove querystring/fragmento
        name = re.split(r"[\\/]", str(c))[-1]
        name = name.split("?")[0].split("#")[0]
        return name or "desconhecido"
    return "desconhecido"

def _page_to_int_one_based(p) -> int | None:
    """Converte 'page' para inteiro 1-based; retorna None se não for possível."""
    try:
        return int(p) + 1
    except Exception:
        return None

def _format_pages(pages: set[int]) -> str:
    """Formata páginas como lista de números e intervalos: 1, 3–5, 9."""
    seq = sorted({x for x in pages if isinstance(x, int)})
    if not seq:
        return "—"
    ranges = []
    start = prev = None
    for x in seq:
        if start is None:
            start = prev = x
        elif x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    if start is not None:
        ranges.append((start, prev))
    parts = [f"{a}" if a == b else f"{a}–{b}" for a, b in ranges]
    return ", ".join(parts)

# ======================================================
#                Aplicação Principal
# ======================================================

def main():
    """Função principal da aplicação."""

    # Inicialização do estado
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_feedback" not in st.session_state:
        st.session_state.show_feedback = False
    if "feedback_enviado" not in st.session_state:
        st.session_state.feedback_enviado = False
    if "zero_message" not in st.session_state:
        st.session_state.zero_message = True

    sid = get_or_create_session_id()

    # ======================================================
    #                     Sidebar
    # ======================================================
    with st.sidebar:
        st.markdown("# Ferramentas:")

        with st.popover("ℹ️ Informações", use_container_width=True):
            st.markdown("### Sobre a aplicação")
            st.write("Este assistente virtual responde perguntas com base nos documentos internos da Unimed.")
            st.write("")
            st.markdown(f"**Versão:** `{APP_VERSION}`")
            st.markdown(f"**Backend:** `{BACKEND_URL}`")
            st.markdown("---")
            st.markdown("""
                Caso algo não funcione como esperado, entre em contato com o suporte técnico.
                Não deixe de avaliar esta aplicação.
            """)

        # Verificação do servidor
        if st.button("🔍 Verificar servidor", use_container_width=True):
            placeholder = st.empty()

            with st.spinner("Verificando..."):
                r = check_backend()
                ok = r.get("ok")

                if ok:
                    placeholder.success("✅ Backend disponível")
                    with st.expander("Detalhes"):
                        st.json(r.get("raw", {}))
                else:
                    placeholder.error("❌ Backend indisponível")
                    with st.expander("Detalhes"):
                        st.json(r.get("raw", {}))

            time.sleep(2)
            placeholder.empty()

        st.markdown("---")

        # Opção de usar documentos internos
        st.write("Marque a opção abaixo para perguntas referentes à política interna da Unimed Blumenau.")

        use_docs = st.checkbox(
            "Usar documentos internos?",
            value=True,
            help="""
            Quando ativado, a IA busca informações nos documentos da qualidade para melhorar a resposta.

            Quando desativado, o usuário pode perguntar sobre assuntos gerais, desde que respeite as regras de segurança e código de ética da Unimed.
            """
        )

        st.markdown("---")

        # Botão para apagar conversa
        if st.button("🗑️ Nova conversa", use_container_width=True):
            reinicia_conversa(session_id=sid, rotate_sid=True)

    # ======================================================
    #                     Cabeçalho
    # ======================================================
    render_header()

    # Recupera o nome do usuário
    user_name = st.session_state.get("usuario", "Usuário")
    if "." in user_name:
        user_name = user_name.split(".")[0].capitalize()

    # ======================================================
    #                  Mensagens antigas
    # ======================================================
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # ======================================================
    #                  Entrada do usuário
    # ======================================================
    user_input = st.chat_input("Digite sua pergunta...")

    if user_input:
        st.session_state.zero_message = False

        # Adiciona mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Processa a pergunta
        with st.chat_message("assistant"):
            with st.spinner("Processando..."):
                try:
                    # Verifica se o backend está disponível
                    hb = check_backend()
                    if not hb.get("ok", False):
                        st.warning(UNAVAILABLE_MSG)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": UNAVAILABLE_MSG
                        })
                        registrar_erro(
                            usuario=st.session_state.get("usuario", "anonimo"),
                            session_id=sid,
                            pergunta=user_input,
                            mensagem="Backend indisponível"
                        )
                    else:
                        # Chama o backend
                        resp = call_backend(
                            user_input,
                            session_id=sid,
                            force_rag=bool(use_docs)
                        )

                        # Extrai dados da resposta
                        reply = resp.get("reply", "")
                        sources = resp.get("sources", [])
                        route = resp.get("route", "")
                        confidence = resp.get("confidence", 0.0)
                        reasoning = resp.get("reasoning", "")
                        latency = resp.get("latency_ms", 0)
                        clarifying_questions = resp.get("clarifying_questions", [])

                        # Exibe resposta
                        if route == "clarify":
                            st.markdown(reply.replace("$", "\\$"))
                            if clarifying_questions:
                                st.markdown("**Perguntas para esclarecer:**")
                                for q in clarifying_questions:
                                    st.markdown(f"- {q}")
                        else:
                            st.markdown(reply.replace("$", "\\$") if reply else "_(sem resposta)_")

                        # Adiciona à lista de mensagens
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": reply or ""
                        })

                        # Atualiza histórico de chat (para próximas perguntas)
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": user_input
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": reply
                        })

                        # Limita histórico a 10 mensagens
                        if len(st.session_state.chat_history) > 10:
                            st.session_state.chat_history = st.session_state.chat_history[-10:]

                        # Exibe fontes se houver
                        if sources and route == "rag":
                            st.markdown("---")
                            st.markdown("### 📚 Fontes:")

                            # Agrupa por documento
                            agrupado: dict[str, dict] = defaultdict(
                                lambda: {"nome": None, "paginas": set()}
                            )

                            for src in sources:
                                metadata = src.get("metadata", {}) if isinstance(src, dict) else {}
                                nome = _nome_fonte(metadata)
                                key = nome.casefold()

                                if agrupado[key]["nome"] is None:
                                    agrupado[key]["nome"] = nome

                                pagina = _page_to_int_one_based(metadata.get("page"))
                                if pagina is not None:
                                    agrupado[key]["paginas"].add(pagina)

                            MAX_FONTES_EXIBIR = 5
                            for key in sorted(agrupado.keys(), key=lambda k: agrupado[k]["nome"])[:MAX_FONTES_EXIBIR]:
                                nome = agrupado[key]["nome"]
                                pags = _format_pages(agrupado[key]["paginas"])
                                if pags == "—":
                                    st.markdown(f"- **{nome}**")
                                else:
                                    st.markdown(f"- **{nome}**, páginas {pags}")

                        # Prepara feedback
                        st.session_state["last_question"] = user_input
                        st.session_state["last_answer"] = reply
                        st.session_state["show_feedback"] = True
                        st.session_state["feedback_enviado"] = False

                        # Registra histórico
                        registrar_historico(
                            usuario=st.session_state.get("usuario", "anonimo"),
                            pergunta=user_input,
                            resposta=reply,
                            session_id=sid,
                            latencia=latency,
                            mode=route,
                            generation_info={
                                "confidence": confidence,
                                "reasoning": reasoning
                            },
                            usage_info={},
                            extra={
                                "question_original": user_input,
                                "question_used": user_input,
                                "use_docs": use_docs
                            }
                        )

                except Exception as error:
                    msg = f"❌ Ocorreu um erro ao processar sua pergunta: {str(error)}"
                    st.error(msg)
                    logger.error(f"Erro no processamento: {error}", exc_info=True)

                    registrar_erro(
                        usuario=st.session_state.get("usuario", "anonimo"),
                        session_id=sid,
                        pergunta=user_input,
                        mensagem=str(error)
                    )

    # ======================================================
    #                  Feedback
    # ======================================================
    if st.session_state.get("show_feedback") and not st.session_state.get("feedback_enviado", False):
        with st.expander(label="💬 Deixe seu feedback", expanded=False):
            with st.form("form_feedback", clear_on_submit=True):
                st.markdown("### ✍️ Como foi a resposta?")
                avaliacao = st.radio(
                    "A resposta foi útil?",
                    options=[1, 0],
                    format_func=lambda x: "👍 Sim, foi útil" if x == 1 else "👎 Não ajudou",
                )
                usuario_feedback = st.text_input("Seu nome (opcional):")
                comentario = st.text_area("Comentário (opcional):", max_chars=280)

                submitted = st.form_submit_button("Enviar feedback")
                if submitted:
                    feedback_data = {
                        "timestamp": now_str(),
                        "session_id": sid,
                        "pergunta": st.session_state.get("last_question", ""),
                        "resposta": st.session_state.get("last_answer", ""),
                        "avaliacao": avaliacao,
                        "usuario": usuario_feedback if usuario_feedback else st.session_state.get("usuario", "Anônimo"),
                        "comentario": comentario,
                    }

                    feedback_path = Path("monitoramento/feedback.log")
                    feedback_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(feedback_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")

                    st.success("✅ Obrigado pelo seu feedback!")
                    st.session_state["feedback_enviado"] = True
                    time.sleep(1)
                    st.rerun()


if __name__ == "__main__":
    # Modo de manutenção (comentar as linhas abaixo para ativar o app)
    # st.markdown("""
    #     <div style="border: 2px solid #FFCC00; border-radius: 10px; padding: 20px; background-color: #FFF8E1;">
    #     <h3 style="color: #E65100;">⚠️ Aviso de indisponibilidade temporária</h3>
    #     <p><strong>Atenção!</strong> A ferramenta está temporariamente fora do ar.</p>
    #     <p>Estamos realizando <strong>ajustes e melhorias</strong> com base nos dados de uso e nas
    #     <strong>sugestões enviadas por usuários</strong> para tornar a experiência ainda mais eficiente.</p>
    #     <p>⏳ Agradecemos sua compreensão. Em breve, voltaremos com novidades!</p>
    #     </div>
    #     """, unsafe_allow_html=True)

    # Autenticação de usuários (se necessário)
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Descomente para ativar autenticação
    # if not st.session_state.logged_in:
    #     render_login()
    #     st.stop()

    # Executa aplicação principal
    main()
