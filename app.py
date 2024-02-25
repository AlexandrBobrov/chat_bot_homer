from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from answgen import answer

app = Flask(__name__)
socketio = SocketIO(app)

# Список для хранения последних сообщений
messages = {'user': [], 'bot': []}

@app.route('/')
def index():
    return render_template('index.html', messages=messages)

@socketio.on('message')
def handle_message(msg):
    # Добавляем сообщение в список
    context=''
    for ans, res in zip(messages['user'][-4:], messages['bot'][-4:]):
        context+=ans+'[SEP]'+res+'[SEP]'
    
    messages['user'].append(msg)
       
    # Приветственное сообщение от бота
    bot_response = answer(msg, context)
    messages['bot'].append(bot_response)
    
    # Отправляем сообщение обратно клиенту
    emit('message', {'user': msg, 'bot': bot_response}, broadcast=True)

@socketio.on('clear')
def clear_messages():
    # Очистка хранимых сообщений
    messages['user'] = []
    messages['bot'] = []
    # Отправка пустых сообщений клиентам для обновления интерфейса
    emit('message', {'user': '', 'bot': ''}, broadcast=True)
    
if __name__ == '__main__':
    socketio.run(app, debug=True)