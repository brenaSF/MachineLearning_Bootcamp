from functools import reduce

alunos = [
    {'nome':'Ana', 'nota':7.2},
    {'nome':'Ana', 'nota':8.2},
    {'nome':'Ana', 'nota':9.2},
    {'nome':'Ana', 'nota':6.2},
]


aluno_aprovado = lambda aluno: aluno['nota'] >=7

obter_nota = lambda aluno: aluno['nota']
alunos_aprovados = filter(aluno_aprovado,alunos)

print(alunos)

#para imprimir o valor por meio do filter - tenho q converter o tipo para lista
#filter -  retorna um objeto do tipo filter que é um iterador
print(list(alunos_aprovados))