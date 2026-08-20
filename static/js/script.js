// Assuming this is the beginning of the script.js file

const form = document.getElementById("uploadForm");

form.addEventListener("submit", async function (e) {

	e.preventDefault();

	const data = new FormData();

	const file = document.getElementById("file").files[0];

	data.append("file", file);

	const response = await fetch("/upload", {

		method: "POST",

		body: data

	});

	const result = await response.json();

	document.getElementById("result").textContent =
		JSON.stringify(result, null, 4);

});
