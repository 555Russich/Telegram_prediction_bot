import os
import re
import requests
from bs4 import BeautifulSoup

with open('ChatExport_2021-05-15/messages.html', 'r', encoding='utf-8') as file:
    src = file.read()

soup = BeautifulSoup(src, 'lxml')
finder_graph_problem = 'Can not send graph :('
all_text_class = soup.find_all('div', class_='text')
list_of_text = []
for i in all_text_class:
    i = i.text
    list_of_text.append(i)
    # print(i)
# print(list_of_text)
sorted_list_of_text = []

for i in list_of_text:
    split_of_line = re.split('\n +', i)
    sorted_list_of_text.append(split_of_line[1])
print(sorted_list_of_text)

counter_graph = 0
counter_average_price = 0
list_of_founded_average_prices = []
list_of_tickets = []
for i in sorted_list_of_text:
    finder_average_price = 'Average price of predictions '
    result_of_search = re.match(finder_average_price, i)
    find_ticket = re.findall('^[A-Z]+[A-Z]$', i)
    if result_of_search is not None:
        counter_average_price += 1
        list_of_founded_average_prices.append(i)
        # print(result_of_search)
    if find_ticket != []:
        list_of_tickets.append(find_ticket[0])
    if i == 'Can not send graph :(':
        counter_graph += 1


print(f'counter_graph: {counter_graph}')
print(f'counter_average_price: {counter_average_price}')
# print(list_of_founded_average_prices)
# print(list_of_tickets)


''' Finding average prices and do list of them'''
average_prices_list = []
for i in list_of_founded_average_prices:
    average_price = re.findall(r'[\[][0-9.]*[\]]', i)
    average_price = average_price[0]
    average_price = average_price.replace('[', '')
    average_price = average_price.replace(']', '')
    average_prices_list.append(average_price)
# print(average_prices_list)

divs_reply = soup.find_all('div', class_='reply_to details')
for i in divs_reply:
    div_next_after_reply = soup.find(divs_reply[i])
    print(div_next_after_reply)


# avr_price_divs = soup.find_all('div', text=re.compile('Average price of predictions '))
# list_of_all_avr_prices = []
# for i in avr_price_divs:
#     i = i.text
#     abc = soup.find('div',)
#     # print(i)
#
# all_bodies = soup.find_all('div', class_='body')
# for i in all_bodies[0:100]:
#     # print(i)
#     all_avr_prices_div = soup.find_all('div', re.compile('Average price of predictions '))
#     for k in all_avr_prices_div:
#         print(k.text)


    # list_of_all_avr_prices.append(i)
# print(list_of_all_avr_prices)
# for i in list_of_all_avr_prices:
    # print(i)

# abc = soup.find_all('div', text=re.compile(list_of_all_avr_prices[0]))
# print(abc)