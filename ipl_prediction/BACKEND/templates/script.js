document.getElementById('predictionForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission behavior

    // Get the team codes from input fields, trim spaces, and convert to uppercase for consistency
    const team1 = document.getElementById('team1').value.trim().toUpperCase();
    const team2 = document.getElementById('team2').value.trim().toUpperCase();

    // Basic validation to ensure both team names are provided
    if (!team1 || !team2) {
        alert('Please enter valid team codes.');
        return;
    }

    // Send a POST request to the backend API with the team names
    fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ team1, team2 })
    })    
    .then(response => {
        if (!response.ok) {
            return response.json().then(error => { throw new Error(error.error) });
        }
        return response.json();
    })
    .then(data => {
        if (data.winner) {
            // Display the predicted winner
            document.getElementById('winner').textContent = `Predicted Winner: ${data.winner}`;
            document.getElementById('result').style.display = 'block';  // Show the result div
        } else {
            alert('No winner prediction available.');
        }
    })
    .catch(error => {
        // Display error message
        console.error('Error:', error);
        alert(`Error: ${error.message}`);
    });
});
