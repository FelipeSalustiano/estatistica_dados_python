import joblib
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv('clientes-v3-preparado.csv')

# Categorizar salário: acima e abaixo da mediana 
df['salario_categoria'] = (df['salario'] > df['salario'].median()).astype(int) # 1 - Acima da mediana, 0 - abaixo ou igual a mediana 

X = df[['idade', 'anos_experiencia', 'nivel_educacao_cod', 'area_atuacao_cod']]
Y = df['salario_categoria']

# Dividir dados: Treinamento e Teste 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Criar e treinar modelo de regressão logistica 
modelo_lr = LogisticRegression()
modelo_lr.fit(X_train, Y_train)

# Criar e treinar modelo de Árvore de Decisão
modelo_dt = DecisionTreeClassifier()
modelo_dt.fit(X_train, Y_train)

# Prever os valores de teste
Y_prev_lr = modelo_lr.predict(X_test)
Y_prev_dt = modelo_dt.predict(X_test)

# Métricas de avaliação - Regressão Logística 
accuracy_lr = accuracy_score(Y_test, Y_prev_lr)
precision_lr = precision_score(Y_test, Y_prev_lr)
recall_lr = recall_score(Y_test, Y_prev_lr)

print(f'\nAcurácia da Regressão Logística: {accuracy_lr:.2f}')
print(f'Precisão da Regressão Logística: {precision_lr:.2f}')
print(f'Recall (sensibilidade) da Regressão Logística: {recall_lr:.2f}')

# Métricas de avaliação - Árvore de Decisão
accuracy_dt = accuracy_score(Y_test, Y_prev_dt)
precision_dt = precision_score(Y_test, Y_prev_dt)
recall_dt = recall_score(Y_test, Y_prev_dt)

print(f'\nAcurácia da Árvore de Decisão: {accuracy_dt:.2f}')
print(f'Precisão da Árvore de Decisão: {precision_dt:.2f}')
print(f'Recall (sensibilidade) da Árvore de Decisão: {recall_dt:.2f}')

joblib.dump(modelo_lr, 'modelo_regressão_logistica.pkl')
joblib.dump(modelo_dt, 'modelo_arvore_decisao.pkl')
