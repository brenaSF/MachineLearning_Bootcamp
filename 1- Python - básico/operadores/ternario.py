# possui 3 partes - resultado verdadeiro , expressao logica, resultado quando falso
lockdown = False
grana = 30 
status = 'Em casa' if lockdown or grana <= 100 else 'Uhuuuu'

print(status)

# do python 3
print(f'O status e : {status}')