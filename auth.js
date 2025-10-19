// Login
function login(event) {
    event.preventDefault();
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    if (email=="u5681011@warwick.ac.uk" && password=="hello"){
        window.location.replace("landingpage.html");
        alert("Logged in");
    }
    errorDiv.style.display = 'none';
}

