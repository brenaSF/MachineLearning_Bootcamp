'''
x = 10
while x:
    print(x)
    x-=1

print('Fim!')

'''




total = 0.0
qtde = 0 
nota = 0 
#quantidade indeterminada 
#para ao digitar -1
while nota != -1:
    nota = float(input('Informe o número ou -1 para sair: '))
    if nota != -1:
        qtde +=1
        total += nota

print(f'A média da turma vale {total/qtde}!')



