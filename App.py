import streamlit as st # TO-RUN: streamlit run <filename.py>


st.markdown("# Discover Your Happiness")

redes_options = ['LinkedIn', 'Twitch', 'YouTube', 'Instagram', 'Amigos', 'Google / Browser', 'Twitter / X', 'Outra rede social']
redes = st.selectbox("Como conheceu o Téo?", options=redes_options)

col1, col2, col3 = st.columns(3)

with col1:
    games = st.radio('Curte games?', ["Sim", "Não"])
    futebol = st.radio('Curte futebol?', ["Sim", "Não"])

    estado_opt = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO']
    estado = st.selectbox("Estado que mora atualmente?", options=estado_opt)

    tempo_area_opt = ['De 0 a 6 meses', 'De 6 meses a 1 ano', 'De 1 a 2 anos', 'De 2 a 4 anos', 'Mais de 4 anos', 'Não atuo']
    tempo_area = st.selectbox("Há quanto tempo atua na Área de Dados?", options=tempo_area_opt)

with col2:
    livros = st.radio('Curte livros?', ["Sim", "Não"])
    jogos_de_tabuleiro = st.radio('Curte jogos de tabuleiro?', ["Sim", "Não"])

    cursos_opt = ['0', '1', '2', '3', '+ que 3']
    cursos = st.selectbox("Quantos cursos do Téo acompanhou?", options=cursos_opt)

    area_opt = ['Administração/Gestão', 'Artes', 'Aposentado/Pensionista', 'Ciências da Natureza', 'Direito', 'Engenharias', 'Estatística/Matemática', 'Linguagens', 'Medicina', 'Tecnologia', 'Outras']
    area = st.selectbox("Qual sua área de formação?", options=area_opt)

with col3:
    jogos_de_fórmula_1 = st.radio('Curte jogos de fórmula 1?', ["Sim", "Não"])
    jogos_de_MMA = st.radio('Curte jogos de MMA?', ["Sim", "Não"])
    age = st.number_input('Sua Idade', 18, 100)

    senioridade_opt = ['Júnior', 'Pleno', 'Sênior', 'Gestor', 'Diretor', 'Especialisa / Consultor', 'C-Level']
    senioridade = st.selectbox("Qual sua cadeira/senioridade?", options=senioridade_opt)

