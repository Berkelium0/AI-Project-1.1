import sys, requests, pathlib, json

if len(sys.argv) != 2:
    print('Usage: verify.py path/to/your/example-solutions.csv')
    sys.exit()

with open(sys.argv[1]) as fp:
    data = fp.read()

response = requests.get(f'http://jfschaefer.de:8973/verify/ss24a11/code', data=data)

print(response.text)

print()
print('WARNING: YOU MAY GET A DIFFERENT NUMBER OF POINTS')
print('The script only checks your solutions for the example problems, '
        'but your grade will depend on the actual solutions. '
        'Furthermore, there is a chance that the script contains mistakes. '
        'The script is only intended to help you with debugging.')

