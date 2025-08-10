import streamlit as st
import pandas as pd

# --- Configuração inicial ---
st.set_page_config(page_title="Rotinas e Hábitos", page_icon="✅", layout="centered")
st.title("📅 Gerenciador de Rotinas e Hábitos")

# --- Inicializa os dados na sessão ---
if "tarefas" not in st.session_state:
    st.session_state.tarefas = pd.DataFrame(columns=["Tarefa", "Categoria", "Prioridade", "Status"])
if "mostrar_form" not in st.session_state:
    st.session_state.mostrar_form = False

# --- Botão para abrir pop-up ---
col_add, col_remove = st.columns(2)
if col_add.button("➕ Adicionar nova tarefa"):
    st.session_state.mostrar_form = True

# --- Botão para remover tarefas ---
if col_remove.button("🗑️ Remover tarefas") and not st.session_state.tarefas.empty:
    tarefas_remover = st.multiselect("Selecione tarefas para remover", st.session_state.tarefas["Tarefa"].tolist(), key="remover_popup")
    if st.button("Confirmar remoção"):
        st.session_state.tarefas = st.session_state.tarefas[~st.session_state.tarefas["Tarefa"].isin(tarefas_remover)].reset_index(drop=True)
        st.success("Tarefas removidas!")

# --- Simulação de Pop-up para adicionar ---
if st.session_state.mostrar_form:
    st.markdown("### ✏️ Adicionar Tarefa")
    with st.form("form_tarefa", clear_on_submit=True):
        tarefa = st.text_input("Descrição da tarefa")
        categoria = st.selectbox("Categoria", ["Trabalho", "Estudos", "Pessoal", "Saúde", "Outro"])
        prioridade = st.selectbox("Prioridade", ["Alta", "Média", "Baixa"])
        status = st.selectbox("Status", ["Pendente", "Em andamento", "Concluído"])
        colf1, colf2 = st.columns(2)
        adicionar = colf1.form_submit_button("Adicionar")
        cancelar = colf2.form_submit_button("Cancelar")

        if adicionar:
            if tarefa.strip() != "":
                nova_linha = {"Tarefa": tarefa, "Categoria": categoria, "Prioridade": prioridade, "Status": status}
                st.session_state.tarefas = pd.concat([st.session_state.tarefas, pd.DataFrame([nova_linha])], ignore_index=True)
                st.success("Tarefa adicionada!")
                st.session_state.mostrar_form = False
            else:
                st.warning("Digite uma tarefa antes de adicionar.")
        if cancelar:
            st.session_state.mostrar_form = False

# --- Ordenar tabela ---
st.subheader("Tabela de tarefas")
if not st.session_state.tarefas.empty:
    col1, col2 = st.columns([2, 1])
    with col1:
        ordem = st.selectbox("Ordenar por", st.session_state.tarefas.columns)
    with col2:
        ordem_crescente = st.checkbox("Ordem crescente", value=True)
    if ordem:
        st.session_state.tarefas = st.session_state.tarefas.sort_values(by=ordem, ascending=ordem_crescente, ignore_index=True)

# --- Mostrar tabela final ---
if st.session_state.tarefas.empty:
    st.info("Nenhuma tarefa adicionada ainda.")
else:
    st.dataframe(st.session_state.tarefas, use_container_width=True)
