const codeInput =
    document.getElementById("codeInput");

const charCount =
    document.getElementById("charCount");


// =========================================================
// CHARACTER COUNT
// =========================================================

codeInput.addEventListener(
    "input",
    updateCharacterCount
);


function updateCharacterCount() {

    const length =
        codeInput.value.length;

    if (charCount) {

        charCount.textContent =
            `${length} characters`;
    }
}


updateCharacterCount();


// =========================================================
// ANALYZE CODE
// =========================================================

async function analyzeCode() {

    const code =
        codeInput.value.trim();

    const languageElement =
        document.getElementById("language");

    const language =
        languageElement
            ? languageElement.value
            : "Python";


    if (!code) {

        alert(
            "Please enter some code first."
        );

        return;
    }


    const button =
        document.getElementById(
            "analyzeButton"
        );

    const loading =
        document.getElementById(
            "loading"
        );

    const results =
        document.getElementById(
            "results"
        );


    button.disabled = true;

    button.innerHTML =
        "ANALYZING...";


    if (loading) {

        loading.classList.remove(
            "hidden"
        );
    }


    if (results) {

        results.classList.add(
            "hidden"
        );
    }


    try {

        const response =
            await fetch(
                "/api/analyze",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        code: code,

                        language: language

                    })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Analysis failed."
            );
        }


        if (!data.success) {

            throw new Error(
                data.error ||
                "Analysis failed."
            );
        }


        displayResults(
            data
        );


    } catch (error) {

        console.error(
            "Analysis Error:",
            error
        );

        alert(
            error.message
        );

    } finally {

        button.disabled = false;

        button.innerHTML =
            "<span>⚡</span> ANALYZE CODE";


        if (loading) {

            loading.classList.add(
                "hidden"
            );
        }
    }
}


// =========================================================
// DISPLAY RESULTS
// =========================================================

function displayResults(data) {

    const results =
        document.getElementById(
            "results"
        );


    if (!results) {
        return;
    }


    results.classList.remove(
        "hidden"
    );


    const prediction =
        document.getElementById(
            "prediction"
        );


    const confidence =
        document.getElementById(
            "confidence"
        );


    const riskScore =
        document.getElementById(
            "riskScore"
        );


    const issueCount =
        document.getElementById(
            "issueCount"
        );


    const riskNumber =
        document.getElementById(
            "riskNumber"
        );


    const riskLevel =
        document.getElementById(
            "riskLevel"
        );


    // -----------------------------------------------------
    // Prediction
    // -----------------------------------------------------

    if (prediction) {

        prediction.textContent =
            formatPrediction(
                data.prediction
                    ? data.prediction.label
                    : "No Bug"
            );
    }


    // -----------------------------------------------------
    // Confidence
    // -----------------------------------------------------

    if (confidence) {

        confidence.textContent =
            `${
                data.prediction
                    ? data.prediction.confidence
                    : 0
            }%`;
    }


    // -----------------------------------------------------
    // Risk Score
    // -----------------------------------------------------

    const score =
        Number(
            data.risk_score || 0
        );


    if (riskScore) {

        riskScore.textContent =
            `${score}/100`;
    }


    if (issueCount) {

        issueCount.textContent =
            data.issue_count || 0;
    }


    if (riskNumber) {

        riskNumber.textContent =
            score;
    }


    if (riskLevel) {

        riskLevel.textContent =
            `${data.risk_level || "Safe"} Risk`;
    }


    // -----------------------------------------------------
    // Issues
    // -----------------------------------------------------

    displayIssues(
        data.issues || []
    );


    // -----------------------------------------------------
    // Scroll
    // -----------------------------------------------------

    results.scrollIntoView({
        behavior: "smooth"
    });
}


// =========================================================
// FORMAT PREDICTION
// =========================================================

function formatPrediction(label) {

    const value =
        String(
            label || "No Bug"
        ).toLowerCase();


    if (

        value === "1" ||

        value.includes("bug") ||

        value.includes("fault")

    ) {

        return "Bug Detected";
    }


    return "No Bug";
}


// =========================================================
// DISPLAY ISSUES
// =========================================================

function displayIssues(issues) {

    const container =
        document.getElementById(
            "issuesList"
        );


    if (!container) {
        return;
    }


    if (
        !Array.isArray(issues) ||
        issues.length === 0
    ) {

        container.innerHTML = `

            <div class="issue-item">

                <div class="issue-top">

                    <span class="issue-type">
                        ✓ No Static Issues
                    </span>

                    <span class="severity severity-low">
                        CLEAN
                    </span>

                </div>

                <p>
                    The static analyzer did not
                    find any obvious code issues.
                </p>

            </div>

        `;

        return;
    }


    container.innerHTML =
        issues.map(
            issue => {

                const severity =
                    issue.severity ||
                    "Low";


                const line =
                    issue.line ||
                    1;


                const type =
                    issue.type ||
                    "Code Issue";


                const message =
                    issue.message ||
                    "Potential issue detected.";


                const suggestion =
                    issue.suggestion ||
                    "Review this section of the code.";


                return `

                    <div class="issue-item">

                        <div class="issue-top">

                            <span class="issue-type">

                                ${escapeHtml(
                                    type
                                )}

                            </span>


                            <span class="severity severity-${severity.toLowerCase()}">

                                ${escapeHtml(
                                    severity
                                )}

                            </span>

                        </div>


                        <div class="issue-line">

                            Line ${line}

                        </div>


                        <p>

                            ${escapeHtml(
                                message
                            )}

                        </p>


                        <p>

                            <strong>
                                Suggested Fix:
                            </strong>

                            ${escapeHtml(
                                suggestion
                            )}

                        </p>

                    </div>

                `;
            }
        )
        .join("");
}


// =========================================================
// HTML ESCAPE
// =========================================================

function escapeHtml(value) {

    return String(
        value || ""
    )

        .replace(
            /&/g,
            "&amp;"
        )

        .replace(
            /</g,
            "&lt;"
        )

        .replace(
            />/g,
            "&gt;"
        )

        .replace(
            /"/g,
            "&quot;"
        )

        .replace(
            /'/g,
            "&#039;"
        );
}