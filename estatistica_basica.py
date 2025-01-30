import pandas as pd 
import numpy as np

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('clientes-v3-preparado.csv')

print(df.head())

print('Estatística com pandas:')
print('Média: ', df['salario'].mean())
print('Mediana: ', df['salario'].median())
print('Variância: ', df['salario'].var())
print('Desvio Padrão: ', df['salario'].std())
print('Moda: ', df['salario'].mode()[0])
print('Mínimo: ', df['salario'].min())
print('Máximo: ', df['salario'].max())
print('Quartil: ', df['salario'].quantile([0.25, 0.5, 0.75]))
print('Contagem de não nulos: ', df['salario'].value_counts().sum())
print('Soma: ', df['salario'].sum())

# Estrutura de Dados
print('\nColuna do DataFrame: \n', df['salario'])
print('Array do campo: ', df['salario'].values)

print('Estatística com NumPy')
print('Média coluna: ', np.mean(df['salario']))
print('Média com array: ', np.mean(df['salario'].values))

array_campo = df['salario'].values 
print('Mediana: ', np.median(array_campo))
print('Variância: ', np.var(array_campo))
print('Desvio Padrão: ', np.std(array_campo))
print('Mínimo: ', np.min(array_campo))
print('Quartis: ', np.quantile(array_campo, [0.25, 0.5, 0.75]))
print('Porcentagem 25%, 50% e 75%: ', np.percentile(array_campo, [25, 50, 75]))
print('Máximo: ', np.max(array_campo))
print('Contagem de não zeros: ', np.count_nonzero(array_campo))
print('Soma: ', np.sum(array_campo))
