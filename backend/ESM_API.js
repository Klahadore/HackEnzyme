// ESMAPI.js

// Step 1: Extract the enzyme sequence from the section dynamically
document.addEventListener("DOMContentLoaded", () => {
  const enzymeSequenceSection = document.getElementById("output-prediction");
  const enzymeSequence = enzymeSequenceSection
    ? enzymeSequenceSection.textContent.trim()
    : null;

  // Log the extracted sequence to ensure it's correct
  console.log("Extracted Enzyme Sequence:", enzymeSequence);

  // Step 2: Submit the enzyme sequence to the ESMFold API
  if (enzymeSequence) {
    const apiKey = "your_api_key_here"; // Replace with your actual API key
    const apiUrl = "https://build.nvidia.com/meta/esmfold"; // Replace with actual API endpoint

    // Define the payload for the POST request
    const payload = {
      sequence: enzymeSequence,
    };

    // Submit the POST request using fetch API
    fetch(apiUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Job submitted successfully!", data);

        // Step 3: Assuming the JSON contains the PDB data as a string
        const pdbData = data.pdb; // Modify this based on the actual key for the PDB data in the API response

        // Step 4: Convert the PDB data into a Blob and create a URL
        const pdbBlob = new Blob([pdbData], { type: "text/plain" });
        const pdbUrl = URL.createObjectURL(pdbBlob);

        // Step 5: Use NGL Viewer to display the PDB file from the Blob URL
        const stage = new NGL.Stage("pdb-viewport"); // Attach the viewer to the div with id 'pdb-viewport'
        stage
          .loadFile(pdbUrl, { ext: "pdb" }) // Load the PDB structure from the Blob URL
          .then(function (component) {
            component.addRepresentation("cartoon"); // Choose a suitable representation for the protein structure
            component.autoView(); // Auto zoom to fit the structure in the viewport
          })
          .catch((error) => {
            console.error("Error loading PDB file:", error);
          });
      })
      .catch((error) => {
        console.error("Error submitting job:", error);
      });
  } else {
    console.log("No enzyme sequence found in the specified section.");
  }
});
