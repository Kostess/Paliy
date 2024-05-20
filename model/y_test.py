from requests import get

height = input('Введите рост = ')
weight = input('Введите вес = ')
sex = input('Введите пол = ')
# if sex == 'Мужской':
#     sex = 1
# else:
#     sex = 0
print(get('http://localhost:5000/api_linear',
          json={'height': height,
                'weight': weight,
                'sex': sex}).json())
