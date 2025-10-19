import axios from 'axios';

async function fetchData(){
    fetch("https://newsapi.org/v2/everything?q=keyword&apiKey=6b881885d556401ba7e8ab1bb0707aa7")
    .then(a)
    const data = await response.json();
}

function testSkills(){
    window.location.replace("FactCheck.html");
}