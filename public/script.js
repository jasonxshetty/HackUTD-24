// public/script.js

document
  .getElementById('recommendationForm')
  .addEventListener('submit', async (e) => {
    e.preventDefault();
    console.log('Form submitted');
    const customerName = document.getElementById('customerName').value.trim();

    if (!customerName) {
      displayRecommendations(['Error: Customer name cannot be empty.']);
      return;
    }

    try {
      const response = await fetch(
        `/recommendations?customerName=${encodeURIComponent(customerName)}`
      );
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const recommendations = await response.json();
      displayRecommendations(recommendations);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      displayRecommendations([`Error: ${error.message}`]);
    }
  });

function displayRecommendations(recommendations) {
  const recommendationsList = document.getElementById('recommendations');
  recommendationsList.innerHTML = '';

  if (recommendations.length === 0) {
    recommendationsList.innerHTML = '<li>No recommendations found.</li>';
    return;
  }

  recommendations.forEach((explanation) => {
    const listItem = document.createElement('li');
    listItem.textContent = explanation;
    recommendationsList.appendChild(listItem);
  });
}