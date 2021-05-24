// var socket = new WebSocket('ws://localhost:8000/ws/some_url/');

// socket.onmessage = function(event){
//     var data = JSON.parse(event.data);
//     console.log(data);
//     // document.querySelector('#app').innerText = data.message;
//     document.querySelector('#chat-text').value += (data.message + '\n');
// }

// chatSocket.onmessage = function (e) {
//     const data = JSON.parse(e.data);
//     console.log(data)
//     document.querySelector('#chat-text').value += (data.username + ': ' + data.message + '\n')
// }

// document.querySelector('#submit').onclick = function (e) {
//     const messageInputDom = document.querySelector('#input');
//     const message = messageInputDom.value;
//     chatSocket.send(JSON.stringify({
//         'message': message,
//         'username': user_username,
//     }));
//     messageInputDom.value = '';
// };


const e = document.getElementById("select_model");
const value = e.options[e.selectedIndex].text;
console.log("The inputed model is: ", value);
