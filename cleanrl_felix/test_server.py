import torch


if __name__ == '__main__':
    y = 1
    m_list = [1, 2, 3]

    for x in m_list:
        y*=x

    print(y)

    print("Did it")