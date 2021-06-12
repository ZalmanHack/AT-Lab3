import colorama
from colorama import init, Fore, Back, Style
from termcolor import colored

class custom_console:

    @staticmethod
    def get_max_len(inline):
        return len(max([str(x) for x in inline], key=len)) + 1

    @staticmethod
    def build_row(inline: list, max_width=0):
        colorama.init()
        result_line = ''
        if max_width == 0:
            max_width = custom_console.get_max_len(inline)
        for index, i in enumerate(inline):
            if isinstance(i, float) or isinstance(i, int):
                spaces = max_width - len(str("{:.2f}".format(i)))
                line = str("{:.2f}".format(i)) + ' ' * spaces
                if i == 0:
                    result_line += colored(line, 'white')
                elif i < 25:
                    result_line += colored(line, 'white', 'on_red')
                elif i < 75:
                    result_line += colored(line, 'white', 'on_yellow')
                elif i < 90:
                    result_line += colored(line, 'white', 'on_green')
                else:
                    result_line += colored(line, 'white', 'on_cyan')
            elif isinstance(i, str) and index > 0 and len(i) == 0:
                spaces = max_width - len(str(i))
                result_line += colored(i + ' ' * spaces, 'white', 'on_white')
            else:
                spaces = max_width - len(str(i))
                result_line += colored(i + ' ' * spaces)
        return result_line


if __name__ == "__main__":
    colorama.init()
    print(custom_console.build_row(['Текст.txt', 0, 15, 50, 60, 88], 0))
