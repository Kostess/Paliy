from requests import get

height = input('Введите рост = ')
weight = input('Введите вес = ')
gender = input('Введите пол = ')
if gender == 'м':
    gender = 1
else:
    gender = 0
print(get('http://localhost:5000/api_linear',
          json={'height': height,
                'weight': weight,
                'gender': gender}).json())
