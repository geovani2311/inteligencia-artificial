import streamlit as st
import pandas as pd
import plotly.express as px

# --- Configuração inicial ---
st.set_page_config(page_title="Rotinas e Hábitos", page_icon="✅", layout="wide")
st.title("📅 Gerenciador de Rotinas e Hábitos")

# --- Inicializa os dados na sessão ---
if "tarefas" not in st.session_state:
    st.session_state.tarefas = pd.DataFrame(columns=["Tarefa", "Categoria", "Prioridade", "Status", "Observações"])
if "mostrar_form" not in st.session_state:
    st.session_state.mostrar_form = False
if "mostrar_remover" not in st.session_state:
    st.session_state.mostrar_remover = False
if "mostrar_dashboard" not in st.session_state:
    st.session_state.mostrar_dashboard = False

# --- Barra de botões principais ---
col_add, col_remove, col_dash = st.columns(3)
if col_add.button("➕ Adicionar nova tarefa"):
    st.session_state.mostrar_form = True
    st.session_state.mostrar_remover = False
    st.session_state.mostrar_dashboard = False

if col_remove.button("🗑️ Remover tarefas") and not st.session_state.tarefas.empty:
    st.session_state.mostrar_remover = True
    st.session_state.mostrar_form = False
    st.session_state.mostrar_dashboard = False

if col_dash.button("📊 Ver Dashboard") and not st.session_state.tarefas.empty:
    st.session_state.mostrar_dashboard = True
    st.session_state.mostrar_form = False
    st.session_state.mostrar_remover = False

# --- Pop-up para remover tarefas ---
if st.session_state.mostrar_remover and not st.session_state.tarefas.empty:
    st.markdown("### 🗑️ Remover Tarefas")
    tarefas_remover = st.multiselect("Selecione tarefas para remover", st.session_state.tarefas["Tarefa"].tolist(), key="remover_popup")
    colr1, colr2 = st.columns(2)
    confirmar = colr1.button("Confirmar remoção")
    cancelar_remover = colr2.button("Fechar")

    if confirmar and tarefas_remover:
        st.session_state.tarefas = st.session_state.tarefas[~st.session_state.tarefas["Tarefa"].isin(tarefas_remover)].reset_index(drop=True)
        st.success("Tarefas removidas!")
        st.session_state.mostrar_remover = False
    elif confirmar and not tarefas_remover:
        st.warning("Selecione pelo menos uma tarefa para remover.")
    if cancelar_remover:
        st.session_state.mostrar_remover = False

# --- Pop-up para adicionar tarefas ---
if st.session_state.mostrar_form:
    st.markdown("### ✏️ Adicionar Tarefa")
    with st.form("form_tarefa", clear_on_submit=True):
        tarefa = st.text_input("Descrição da tarefa")
        categoria = st.selectbox("Categoria", ["Trabalho", "Estudos", "Pessoal", "Saúde", "Outro"])
        prioridade = st.selectbox("Prioridade", ["Alta", "Média", "Baixa"])
        status = st.selectbox("Status", ["Pendente", "Em andamento", "Concluído"])
        observacoes = st.text_area("Observações (opcional)")
        colf1, colf2 = st.columns(2)
        adicionar = colf1.form_submit_button("Adicionar")
        cancelar = colf2.form_submit_button("Fechar")

        if adicionar:
            if tarefa.strip() != "":
                nova_linha = {"Tarefa": tarefa, "Categoria": categoria, "Prioridade": prioridade, "Status": status, "Observações": observacoes}
                st.session_state.tarefas = pd.concat([st.session_state.tarefas, pd.DataFrame([nova_linha])], ignore_index=True)
                st.success("Tarefa adicionada!")
                st.session_state.mostrar_form = False
            else:
                st.warning("Digite uma tarefa antes de adicionar.")
        if cancelar:
            st.session_state.mostrar_form = False

# --- Dashboard com Plotly Express ---
if st.session_state.mostrar_dashboard and not st.session_state.tarefas.empty:
    st.markdown("## 📊 Dashboard de Rotinas")

    # Corrigindo para evitar erro com value_counts().reset_index()
    df_cat = st.session_state.tarefas["Categoria"].value_counts().reset_index()
    df_cat.columns = ["Categoria", "Quantidade"]
    fig_cat = px.bar(df_cat, x="Categoria", y="Quantidade", title="Tarefas por Categoria")
    st.plotly_chart(fig_cat, use_container_width=True)

    fig_status = px.pie(st.session_state.tarefas, names="Status", title="Distribuição por Status")
    st.plotly_chart(fig_status, use_container_width=True)

    df_prioridade = st.session_state.tarefas["Prioridade"].value_counts().reset_index()
    df_prioridade.columns = ["Prioridade", "Quantidade"]
    fig_prioridade = px.bar(df_prioridade, x="Prioridade", y="Quantidade", title="Tarefas por Prioridade")
    st.plotly_chart(fig_prioridade, use_container_width=True)

# --- Ordenar tabela ---
if not st.session_state.tarefas.empty and not st.session_state.mostrar_dashboard:
    st.subheader("Tabela de tarefas")
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
elif not st.session_state.mostrar_dashboard:
    st.dataframe(st.session_state.tarefas, use_container_width=True)
