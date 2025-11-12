# ////////////////////////////////////////////////////////////////////////////////////////
#
# Chat_RAG Pipeline
#
# Unimed Blumenau
#
# Criado por Breno Andrado / Robson Hostin
#
# Descrição: Cria o pipeline de carga dos documentos incrementar de documentos para o banco
#            de dados vetorizado do Pinecone para criação do RAG para o Chat.
#            Utiliza bibliotecas do landchain para extração dos textos dos arquivos e a
#            criação dos chunk para envio ao Pinecone.
#            A rotina irá Buscar os documentos da Função Gestão da Qualidade, ainda não enviados
#            ou que tiveram alteração para atualização.
#
# Estrutura: 1) Importação das bibliotecas;
#            2) Conexões com bando de dados
#            3) Funções de apoio;
#            4) Extração do texto dos documentos, chunks e envio para bando de dados vetorizado
#            5) Carga dos documentos para processamento.
#            6) Cria documento PDF com lista de todos documentos da qualidade
#            7) Faz limpeza da pasta temporaria
#
# ////////////////////////////////////////////////////////////////////////////////////////

# -----------------------------------------------------------------------------------------
# 1) BIBLIOTECAS
# -----------------------------------------------------------------------------------------

import os
import re
import subprocess
import hashlib
import unicodedata
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import difflib
from dotenv import load_dotenv

#Bibliotecas para banco vetorizado
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException

#Bibliotecas para leitura e chunk dos documentos
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

#Bibliotecadas para conexao com o banco de dados
from sqlalchemy import create_engine, text, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import IntegrityError
import cx_Oracle

load_dotenv()

# -----------------------------------------------------------------------------------------
# 2) CONFIGURAÇÕES E CONSTANTES
# -----------------------------------------------------------------------------------------

class Config:
    """Classe para armazenar as configurações do pipeline"""

    # Caminho absoluto para a pasta Conversao_documentos
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CONVERSAO_DIR = BASE_DIR + "/conversao_documentos"
    ARQUIVO_LOG = os.path.join(BASE_DIR, "log", "log_execucao.txt")

    # Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")

    # PINECONE
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_DSUNIBLU")
    INDEX_NAME = os.getenv("PINECONE_INDEX", "pinecone-vector-store-optimized")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

    #Mapeamento dos caminhos Windows (SMB) para caminho Linux montado
    MAPPINGS = {
        r"\\blumenau.unimed\dfs\APPS": "/mnt/APPS",
        r"\\blumenau.unimed\dfs\Qualidade": "/mnt/Qualidade",
        r"R:\Qualidade": "/mnt/Qualidade"
    }

    # Chunking otimizado
    CHUNK_CONFIGS = {
        'small': {'size': 500, 'overlap': 100},
        'medium': {'size': 1000, 'overlap': 200},
        'large': {'size': 1800, 'overlap': 300},
        'parent': {'size': 2500, 'overlap': 500}
    }

class TipoDocumento(Enum):
    """Tipos de documentos suportados pelo sistema de qualidade"""
    MANUAL = "MAN"
    DIRETRIZ = "DIR"
    INSTRUCAO_TRABALHO = "IT"
    FORMULARIO = "FOR"
    DOCUMENTO_EXTERNO = "DE"


# -----------------------------------------------------------------------------------------
# 3) CONEXÕES COM BANDO DE DADOS
# -----------------------------------------------------------------------------------------
try:
    cx_Oracle.init_oracle_client(os.getenv('ORA_INSTANT_CLIENTE'))
except:
    pass

oracle_url = os.getenv("URL_ORACLE_DB")
engine = create_engine(oracle_url)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class Documentos_Enviados(Base):
    __tablename__ = 'DS_RAG_DOCUMENTOS'
    __table_args__ = {'schema': 'DATASCIENCE'}

    cd_documento = Column(String, primary_key=True)
    id_doc_rag = Column(String)
    dt_envio = Column(DateTime, default=datetime.now)
    qt_chunk = Column(Integer)
    ie_status = Column(String)
    ds_erro = Column(Text)


def grava_envio_documento(cd_documento: str, id_doc_rag: str = None, qt_chunk: Integer = None, ie_status: str = 'OK', ds_erro: str = None):
    """Grava registro dos documentos processados no banco de dados para monitoramento"""
    session = Session()
    try:
        session.add(Documentos_Enviados(
                        cd_documento=cd_documento,
                        id_doc_rag=id_doc_rag,
                        qt_chunk=qt_chunk,
                        ie_status=ie_status,
                        ds_erro=ds_erro
                        )
                    )
        session.commit()
    except IntegrityError:
        session.rollback()
        gerar_log(f"[ERRO] Falha ao gravar registro no banco de dados")
    session.close()


# -----------------------------------------------------------------------------------------
# 4) FUNÇÕES DE APOIO
# -----------------------------------------------------------------------------------------

class Logger:
    """Classe simples de gerenciamento de logs centralizados"""

    @staticmethod
    def log(message: str, level: str = "INFO"):
        """Grava mensagem de log no console e em arquivo"""
        try:
            hora = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            log_message = f"{hora} [{level}] {message}"
            print(log_message)  # Imprime no console

            # Grava no arquivo de log
            os.makedirs(os.path.dirname(Config.ARQUIVO_LOG), exist_ok=True)
            with open(Config.ARQUIVO_LOG, 'a', encoding="utf-8-sig") as arquivo:
                arquivo.write(log_message + "\n")
        except Exception as e:
            print(f"[LOGGER-ERROR] {e}")

def gerar_log(message: str, level: str = "INFO"):
    """Grava logs de execução"""
    Logger.log(message, level)


def chunk_id(doc_id: str, i: int) -> str:
    # IDs ASCII, curtos e determinísticos; 512 chars é o limite duro do Pinecone
    return f"{doc_id}-c{str(i).zfill(5)}"[:128]


def convert_doc_to_pdf(input_path: str) -> str:
    """
    Converte um arquivo .doc para .docx usando o LibreOffice em modo headless.
    Retorna o caminho do arquivo convertido.
    """

    # Caminho completo do executável soffice no Windows C:\Program Files\LibreOffice\program\soffice.exe
    soffice_path = 'soffice'

    # Cria a pasta caso não exista
    os.makedirs(Config.CONVERSAO_DIR, exist_ok=True)

    # Executa o LibreOffice em modo headless
    subprocess.run([
        soffice_path,
        "--headless",
        "--convert-to", "pdf",
        "--outdir", Config.CONVERSAO_DIR,
        input_path
    ], check=True)

    # Nome base do arquivo convertido
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(Config.CONVERSAO_DIR, base_name + ".pdf")

    return output_path





def converte_path_to_linux(win_path: str) -> str:
    """
    Converte caminho Windows (SMB) para caminho Linux montado.
    """
    # Normaliza barras
    path = win_path.replace("\\", "/")

    for win_prefix, linux_prefix in Config.MAPPINGS.items():
        if path.startswith(win_prefix.replace("\\", "/")):
            return path.replace(win_prefix.replace("\\", "/"), linux_prefix, 1)

    return path  # se não achar prefixo, retorna como está

# -----------------------------------------------------------------------------------------
# 3.1) FUNÇÕES PARA TRATAMENTO DO TEXTO
# -----------------------------------------------------------------------------------------

def linhas_similares(l1, l2, cutoff=0.85):
    """Retorna True se duas linhas forem parecidas o suficiente."""
    return difflib.SequenceMatcher(None, l1.strip(), l2.strip()).ratio() >= cutoff


def remover_cabecalho_rodape(pages, max_linhas=15, cutoff=0.85):
    """
    Remove cabeçalhos (da 2ª página em diante) e rodapés (de todas as páginas),
    comparando páginas linha a linha.
    """
    resultado = []
    prev_lines = []
    prim_cab = 0 #Controla a primeira pagina em que identificou cabeçalho ou rodapé

    for i, page in enumerate(pages):
        linhas = page.page_content.splitlines()

        # Caso ainda não tiver identificado nenh
        if prim_cab == 0 and i < len(pages) - 1:
            prev_lines = pages[i+1].page_content.splitlines()

        # --- Detecta cabeçalho (a partir da 2ª página) ---
        if i > 0 and prev_lines:
            cabecalho = []
            for l_atual, l_prev in zip(linhas[:max_linhas], prev_lines[:max_linhas]):
                if linhas_similares(l_atual, l_prev, cutoff):
                    cabecalho.append(l_atual)
                    prim_cab += 1
                else:
                    break
            if cabecalho:
                linhas = linhas[len(cabecalho):]

        # --- Detecta rodapé (em todas as páginas) ---
        if prev_lines:
            rodape = []
            for l_atual, l_prev in zip(reversed(linhas[-max_linhas:]), reversed(prev_lines[-max_linhas:])):
                x=0
                if linhas_similares(l_atual, l_prev, cutoff):
                    rodape.insert(0, l_atual)
                    prim_cab += 1
                else:
                    break
            if rodape:
                linhas = linhas[:-len(rodape)]

        # monta novo Document preservando metadata
        novo_doc = Document(
            page_content="\n".join(linhas).strip(),
            metadata=page.metadata
        )
        resultado.append(novo_doc)

        prev_lines = page.page_content.splitlines()

    return resultado


def ajustar_quebras_linha(pages):
    """
    Recebe as paginas dos documento e ajusta as quebras de linhas para
    quando a linha atual não terminar com pontuação ".!?;:" e o primeiro caracter
    da proxima linha for minusculo, então remove a quebra de linha  .
    """
    docs_ajustados = []

    for page in pages:
        texto = page.page_content
        linhas = texto.splitlines()
        resultado = []

        for i, linha in enumerate(linhas):
            linha = linha.strip()

            if i > 0:  # existe linha anterior
                anterior = resultado[-1]

                # condição: anterior não termina em pontuação forte
                # e linha atual começa com minúscula
                if anterior and not re.search(r'[.!?;:]\s*$', anterior) and linha and linha[0].islower():
                    if anterior.endswith('-'):  # quebra com hífen
                        resultado[-1] = anterior[:-1] + linha
                    else:
                        resultado[-1] = anterior + " " + linha
                    continue

            resultado.append(linha)

        texto_final = "\n".join(resultado)

        # Adiciona como Document mantendo o metadata original
        docs_ajustados.append(Document(page_content=texto_final, metadata=page.metadata))

    return docs_ajustados

# -----------------------------------------------------------------------------------------
# 4) EXTRAÇÃO DO TEXTO DOS DOCUMENTOS, CHUNKS E ENVIO PARA BANCO DE DADOS VETORIZADO
# -----------------------------------------------------------------------------------------
class PineconeStore:
    """Classe para gerenciar o armazenamento de vetores no Pinecone"""


    def __init__(self):
        api_key = Config.PINECONE_API_KEY
        if not api_key:
            raise ValueError("Variável de ambiente PINECONE_API_KEY não configurada")

        self.pc = Pinecone(api_key=api_key)

        # Cria índice se não existir
        if Config.INDEX_NAME not in self.pc.list_indexes().names():
            gerar_log(f"Índice '{Config.INDEX_NAME}' não existe; criando…")
            try:
                self.pc.create_index(
                    name=Config.INDEX_NAME,
                    dimension=Config.EMBED_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=Config.PINECONE_CLOUD, region=Config.PINECONE_REGION),
                )
            except PineconeApiException as e:
                # 409 = already exists (race condition)
                if getattr(e, "status", None) != 409:
                    raise
        else:
            gerar_log(f"Índice '{Config.INDEX_NAME}' já existe.")

        self.index = self.pc.Index(Config.INDEX_NAME)
        self.embedder = OllamaEmbeddings(model=Config.EMBEDDING_MODEL, base_url=Config.OLLAMA_BASE_URL)


    # -------- Atualização segura (delete + upsert) --------
    def upsert_pdf(
            self,
            file_path:            str,
            file_extension:       str,
            document_id:          str,
            document_name:        str,
            cd_setores_liberados: str,
            namespace:            str  = "default",
            delete_before:        bool = True,
            batch_size:           int  = 100,
            chunk_size:           int  = 2500,
            chunk_overlap:        int  = 100,
            separators:           list = ['\n\n','\n','.']
            ):
        # 1) Carregar o arquivo conforme a extensão
        if file_extension == ('pdf'):
            loader = PyPDFLoader(file_path)
        elif file_extension == ('docx'):
            loader = Docx2txtLoader(file_path)
        elif file_extension == ('doc'):
            # Não é possível fazer a leitura dos arquivos .doc no langchain, é necessário converter para PDF
            file_path = convert_doc_to_pdf(file_path)
            loader = PyPDFLoader(file_path)
        elif file_extension.lower() in ("md", "markdown"):
            loader = TextLoader(file_path, encoding="utf-8")

        pages = loader.load()

        # Rmove cabeçalhos e rodapés das paginas intermediárias
        pages = remover_cabecalho_rodape(pages)
        pages = ajustar_quebras_linha(pages)

        # 1.1) Divide o arquivo
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
        #splitter = RecursiveCharacterTextSplitter(**{"chunk_size": 30, "chunk_overlap": 10 })
        docs = splitter.split_documents(pages)
        #docs = splitter.split_text(pages)

        if not docs:
            grava_envio_documento(document_id, ie_status='WARN', ds_erro='Sem texto extraído de: {file_path}')
            gerar_log(f"[WARN] Sem texto extraído de: {file_path}")
            return

        texts: List[str] = [d.page_content for d in docs]
        doc_id = document_id

        # 2) (Opcional) Apagar vetores antigos desse PDF por filtro
        #if delete_before:
            # metadado 'doc_id' é usado como alvo do filtro
            #self.index.delete(filter={"doc_id": {"$eq": doc_id}}, namespace=namespace)

        # 3) Embeddings
        embeddings = self.embedder.embed_documents(texts)

        if len(embeddings[0]) != Config.EMBED_DIM:
            raise ValueError(
                f"Dimensão do embedding ({len(embeddings[0])}) != EMBED_DIM ({Config.EMBED_DIM}). "
                f"Confirme o modelo e o dimension do índice."
            )

        # 4) Preparar vetores (IDs ASCII seguros)
        vectors = []
        for i, emb in enumerate(embeddings):
            vid = chunk_id(doc_id, i)
            meta = {
                "doc_id": doc_id,                  # usado para update/delete por filtro
                "source": document_id + " - " + document_name, #os.path.abspath(file_path),
                "setores": [x.strip() for x in cd_setores_liberados.split(",")],
                "page": docs[i].metadata.get("page") or 0,
                "text": texts[i],
            }
            vectors.append({"id": vid, "values": emb, "metadata": meta})

        # 5) Upsert em lotes
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)

        grava_envio_documento(document_id, id_doc_rag=doc_id, qt_chunk=len(vectors))
        gerar_log(f"[OK] {len(vectors)} chunks upsertados (namespace='{namespace}', doc_id='{doc_id}').")

# -----------------------------------------------------------------------------------------
# 5) CARGA DOS DOCUMENTOS PARA PROCESSAMENTO
# -----------------------------------------------------------------------------------------
if __name__ == "__main__":

    gerar_log('Incio da Execução')

    store = PineconeStore()

    # Busca os documentos não enviado ou com Alteração da Função Gestao da Qualidade do Tasy
    sql =   """select   a.cd_documento,
                        a.nm_documento,
                        tasy.converte_path_storage_web(a.ds_arquivo) ds_arquivo,
                        NVL((select LISTAGG(l2.CD_SETOR_ATENDIMENTO, ',')
                           from tasy.QUA_DOC_LIB l2
                          where l2.nr_seq_doc = a.nr_sequencia),0) cd_setores_liberados
                 from   tasy.QUA_DOCUMENTO a
                 left join (select  CD_DOCUMENTO,
                                    MAX(DT_ENVIO) DT_ENVIO
                              from  datascience.ds_rag_documentos
                          group by CD_DOCUMENTO) e on e.CD_DOCUMENTO = a.CD_DOCUMENTO
                where 1=1
                  and (a.dt_atualizacao > e.dt_envio or e.dt_envio is null) --Buscas somente os Documentos que tiveram alteração no tasy
                  and   a.ie_situacao = 'A'
                  and regexp_substr(tasy.converte_path_storage_web(a.ds_arquivo),'[^.]+$',1,1) in ('pdf','docx','doc')
                  and a.cd_estabelecimento not in (381) --Manter arquivos do HSC temporariamente ser carregar

                  --CONDIÇÕES TESTES
                  -- and a.cd_documento in ('DIR-306','DIR-070','DIR-246','DIR-073','DIR-062')
                  -- and a.cd_documento in ('DE-014','DE-015','DE-016','DE-021','DIR-075','DIR-140','DIR-177','DIR-260','FOR-382','DIR-271')
                  -- and (e.dt_envio is null or e.dt_envio <= to_date('01/10/2025','DD/MM/YYYY')) --Reenviar todos para remover cabeçalhos
                  -- and rownum <= 10
                  -- and 1=2
            """

    #Inicia conexão com banco de dados
    with engine.connect() as conn:
        result = conn.execute(text(sql))

        for row in result:
            cd_documento = row.cd_documento
            nm_documento = row.nm_documento
            cd_setores_liberados = row.cd_setores_liberados

            #Deifnir Path para rodar em windows ou no Linux
            if os.name == 'nt':
                ds_arquivo = row.ds_arquivo.strip()
            else:
                ds_arquivo = converte_path_to_linux(row.ds_arquivo.strip())

            #gerar_log(f'Processando Arquivo: {row.ds_arquivo.strip()} ==> {ds_arquivo}')

            nome_arquivo = os.path.basename(ds_arquivo)  # extrai só o nome do arquivo
            _, extensao = os.path.splitext(ds_arquivo) # extrai a extensão do arquivo
            extensao = extensao.lstrip(".").lower()

            if not os.path.isfile(ds_arquivo):
                grava_envio_documento(cd_documento, ie_status='WARN', ds_erro='Arquivo não localizado')
                gerar_log(f"[WARN] Arquivo não localizado: {nome_arquivo}")
                continue

            # pule não-PDFs
            if extensao not in ('pdf','docx','doc'):
                grava_envio_documento(cd_documento, ie_status='WARN', ds_erro='Extensão do arquivo inválida')
                gerar_log(f"[WARN] Extensão do arquivo inválida: {nome_arquivo}")
                continue

            try:
                store.upsert_pdf(
                    ds_arquivo,
                    file_extension = extensao,
                    document_id = cd_documento,
                    document_name = nm_documento,
                    cd_setores_liberados = cd_setores_liberados,
                    namespace = "default",
                    delete_before = True,
                    chunk_size=1800,
                    chunk_overlap=300
                    )
            except Exception as e:
                #grava_envio_documento(cd_documento, ie_status='ERRO', ds_erro=f'{nome_arquivo}: {e}')
                #gerar_log(f"[ERRO] {nome_arquivo}: {e}")
                raise(e)

# -----------------------------------------------------------------------------------------
# 6) CRIA DOCUMENTO PDF COM LISTA DOS DOCUMENTOS DA FUNÇÃO GESTAO DA QUALIDADE
# -----------------------------------------------------------------------------------------
    gerar_log('Gerando Lista de Documentos')

    sql =   """select a.cd_documento ||' - '|| a.nm_documento as "Nome do Documento",
                      t.ds_tipo_doc as "Tipo do Documento",
                      s.ds_setor_atendimento as "Setor Responsável"
                 from tasy.QUA_DOCUMENTO a
                 left join tasy.QUA_TIPO_DOC t on t.nr_sequencia = a.NR_SEQ_TIPO
                 left join tasy.setor_atendimento s on s.cd_setor_atendimento = a.cd_setor_atendimento
                where 1=1
                  and a.ie_situacao = 'A'
                  --and 1=2
                """

    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()

    # Monta conteúdo Markdown
    lines = ["# Lista de Documentos Ativos\n"]
    for row in rows:
        lines.append(f"## Documento: {row[0]}")
        lines.append(f"- **Tipo do Documento:** {row[1] or 'N/A'}")
        lines.append(f"- **Setor Responsável:** {row[2] or 'N/A'}")
        lines.append("\n---\n")

    # Salva em arquivo
    ds_arquivo = Config.CONVERSAO_DIR + "/Lista_Documentos.md"
    with open(ds_arquivo, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 7. Envia Lista de Documentos para banco vetorizado
    try:
        store.upsert_pdf(
            ds_arquivo,
            file_extension = 'md',
            document_id = 'Lista_Documentos',
            document_name ='Lista_Documentos',
            cd_setores_liberados = '0',
            namespace = "default",
            delete_before = True)
    except Exception as e:
        grava_envio_documento('Lista_Documentos', ie_status='ERRO', ds_erro=f'{e}')

# -----------------------------------------------------------------------------------------
# 7) LIMPEZA DOS DOCUMENTOS GERADOS E CONVERTIDOS
# -----------------------------------------------------------------------------------------

    #Ao termino limpa o diretório dos documentos convertidos
    p = Path(Config.CONVERSAO_DIR)
    for arquivo in p.iterdir():
        if arquivo.is_file():
            arquivo.unlink()

    gerar_log('Fim da Execução')

# -----------------------------------------------------------------------------------------

    # Upload um arquivo especifico
    # store.upsert_pdf(
    #     file_path="downloads/TESTEDOC.doc",
    #     namespace="default",
    #     delete_before=False,
    #     file_extension='doc',
    #     document_id='FOR-1696',
    # )


    # # Upload de todos os arquivos numa pasta
    # path = "downloads"
    # for fname in os.listdir(path):
    #     fpath = os.path.join(path, fname)
    #     if not os.path.isfile(fpath):
    #         continue
    #     # pule não-PDFs
    #     if not fname.lower().endswith(".pdf"):
    #         continue
    #     try:
    #         store.upsert_pdf(fpath, namespace="default", delete_before=False)
    #     except Exception as e:
    #         print(f"[ERRO] {fname}: {e}")
