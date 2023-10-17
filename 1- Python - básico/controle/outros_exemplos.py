

pessoas = ['Gui','Rebeca']
adj = ['Sapeca','Inteligente']


for p in pessoas:
    for a in adj:
        print(f'{p} é {a}!')

# herança e classe
for i in [1,2,3]:
    pass

for i in range(1,11):
    if i % 2 == 1:
        continue
    print(i,end=' ')


for i in range(1,11):
    if i == 5:
        break
    print(i)

print('Fim')