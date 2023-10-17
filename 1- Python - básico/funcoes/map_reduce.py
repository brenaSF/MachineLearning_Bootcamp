from functools import reduce 

# como funciona o map
def somar_nota(delta):
    def somar(nota):
        return nota + delta
    return somar

notas = [6.4,7.2,5.8,8.4]

notas_finais = map(somar_nota(1.5),notas)
print(notas_finais)


def somar(a,b):
    return a + b

total = reduce(somar,notas,0)
print(total)

# reduce 0 acumulador - realizar o cálculo 
#for i, nota in enumerate(notas):
 #   notas[i] = nota + 1.5

#for i in range(len(notas)):
 #   notas[i] = notas[i] + 1.5

