for i in range(1,11):
    print(i,end=' ')

print('')

#definir início, fim, passo 

for i in range(1,100,7):
   print(i,end=' ')
print('')


nums = [2,4,6,8]

for n in nums: 
    print(n,end= ' ')
print('')

texto = 'Python é muito massa!'

for letra in texto:
   print(letra,end =' ')
print('')

#set - procurar definição 
for n in {1,2,3,4,4,4} :
    print(n,end =' ')


produto = {
    'nome': 'Caneta',
    'preco': 8.80,
    'desc': 0.5
}
print('')

#dicionário
for atrib in produto:
   print(atrib, '==>',produto[atrib],end=' ')

print('')

for atrib,valor in produto.items():
    print(atrib, '==>',produto[atrib],end=' ')
print('')

for valor in produto.values():
    print(valor, end = ' ')
print('')

for atrib in produto.keys():
    print(atrib, end = ' ')