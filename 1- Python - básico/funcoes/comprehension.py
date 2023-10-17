from functools import reduce

alunos = [
    {'nome':'Ana', 'nota':7.2},
    {'nome':'Pedro', 'nota':8.2},
    {'nome':'Brena', 'nota':9.2},
    {'nome':'Rafael', 'nota':6.2},
]


somar = lambda a,b: a+b

alunos_aprovados = [aluno for aluno in alunos if aluno['nota'] >= 7 ]
notas_alunos_aprovados = [aluno['nota'] for aluno in alunos_aprovados]
total = reduce(somar,notas_alunos_aprovados,0)
print(total/len(alunos_aprovados))