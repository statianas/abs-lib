import re
import numpy as np
import string


class BoolVar:
    def __init__(self, value):
        self.value = value
        # print("INIT =", value)

    # '-' — возражения "нет"
    def __neg__(self):
        return BoolVar(not self.value)

    # '+' — дизъюнкция "или"
    def __add__(self, other):
        return BoolVar(self.value or other.value)

    # '*' — конъюнкция "и"
    def __mul__(self, other):
        return BoolVar(self.value and other.value)

    # '>' — импликация "если ..., тогда"
    def __gt__(self, other):
        return BoolVar((not self.value) or other.value)

    # '=' — эквивалентность "ровно"
    def __eq__(self, other):
        return BoolVar((self.value and other.value) or (not self.value and not other.value))

    # строковое представление значения
    def __str__(self):
        return "True" if self.value else "False"

    def __format__(self, format_spec):
        return format(str(self), format_spec)

def vector_of_quants (infunc, size):
    # в питоне знак эквивалентности - это '==', так что заменяем
    infunc = infunc.replace("=", "==")
    mas = string.ascii_letters
    for i in range(size):
        infunc = infunc + ' ' + '+' + ' ' + mas[i] + ' ' + '*' + ' ' + '-' + ' ' + mas[i]
    infunc = '(' + infunc + ')'
    # находим переменные в функции, т.е. просто буквы
    # set() делает этот набор уникальным, ну и сортируем
    sorted(set(re.findall(r"[A-Za-z]", infunc)))
    # или так, если надо без использования регулярных выражений
    variables = sorted(set([c for c in infunc if c.isalpha()]))

    # просто красивое оформление для таблицы
    header = [""] * 2
    for key in variables:
        header[0] += "-" * 7 + "+"
        header[1] += f"   {key}   |"
    header[0] += "-+" + "-" * 7
    header[1] += " | Result"
    #print("\n".join(header + header[0:1]))
    ans = []
    vars_for_eval = {}
    # вариантов входных значений для таблицы - 2 в степени кол-ва переменных
    for variant in range(1 << len(variables)):
        # заполняем входной словарь c представлением переменных
        # в виде экземпляров нашего класса для функции eval()
        # key идут в прямом порядке, а i - в обратном
        for i, key in reversed(list(enumerate(reversed(variables)))):
            # используем биты этого числа для инициализыции булевых значений
            vars_for_eval[key] = BoolVar(variant & (1 << i))
            # вывод строки таблицы истинности
            #print(f" {vars_for_eval[key]:<5}", end=" |")
        # вычисляем результат
        result = eval(infunc, {}, vars_for_eval)
        if result.value != 0: ans.append(1)
        else: ans.append(0)
        #print(f" | {result:<5}")
    ans = np.array(ans)
    return ans
