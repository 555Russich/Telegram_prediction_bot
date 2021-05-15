import telebot

token = '1800443246:AAFd_C_We9gLKneNM9aD8dFQyKocXG5yhrI'
bot = telebot.TeleBot(token)
a = bot.get_me()
print(a)
bot.polling()