# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                               model_definition.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                                romulopauliv@bk.ru ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
from colorama import Fore, Style
# +--------------------------------------------------------------------------------------------------------------------|


def image_reading(name: str) -> None:
    print('>>> image reading ' + Fore.MAGENTA + name + Fore.CYAN + '||| ULTRASOUND: ' + Fore.GREEN, end='')


def ultra_sound_confirm() -> None:
    print('READ   ' + Fore.CYAN + 'MASK: ' + Fore.GREEN, end='')


def mask_confirm() -> None:
    print('READ' + Style.RESET_ALL)



