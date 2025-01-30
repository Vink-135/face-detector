from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/analyze-loan', methods=['POST'])
def analyze_loan():
    try:
        data = request.get_json()
        
        # Validate data
        if not data:
            return jsonify({"error": "Invalid input"}), 400
        
        assets = float(data.get("assets", 0))
        income = float(data.get("income", 0))
        existing_loans = float(data.get("existing_loans", 0))

        # Calculate loan approval probability
        approval_chance = (income * 0.6 + assets * 0.3) - (existing_loans * 0.5)
        approval_probability = max(0, min(100, approval_chance))  # Ensure between 0 and 100

        # Determine recommendation and department redirection
        if approval_probability > 70:
            recommendation = "Highly Recommended"
            department = "Loan Department"
        elif approval_probability > 40:
            recommendation = "Moderate Chance"
            department = "Financial Advisor"
        else:
            recommendation = "Not Recommended"
            department = "Savings & Investment Department"

        return jsonify({
            "approval_probability": round(approval_probability, 2),
            "recommendation": recommendation,
            "redirect_to": department
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)