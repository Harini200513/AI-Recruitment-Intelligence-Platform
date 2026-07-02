import logging
from typing import List, Dict, Any
import pandas as pd

# Setup Logging
logger = logging.getLogger(__name__)

# Predefined Question Bank for common technologies/skills
QUESTION_BANK: Dict[str, Dict[str, List[str]]] = {
    "python": {
        "Easy": [
            "What is the difference between a list and a tuple in Python?",
            "Explain list comprehensions and write a simple example."
        ],
        "Medium": [
            "How does Python's memory management work, and what is the role of the garbage collector?",
            "Explain the difference between deep copy and shallow copy in Python."
        ],
        "Hard": [
            "What are decorators in Python? Write a custom decorator that measures the execution time of a function.",
            "Explain the Global Interpreter Lock (GIL) and how it affects multi-threaded applications."
        ]
    },
    "sql": {
        "Easy": [
            "What is the difference between WHERE and HAVING clauses in SQL?",
            "What are the primary types of JOINs in SQL?"
        ],
        "Medium": [
            "Explain the difference between a clustered and a non-clustered index and how they impact query performance.",
            "What are window functions in SQL? Give an example using ROW_NUMBER() or DENSE_RANK()."
        ],
        "Hard": [
            "How would you optimize a slow-running query that involves multiple tables and millions of records?",
            "Explain transaction isolation levels and the types of read phenomena (dirty reads, non-repeatable reads, phantom reads) they prevent."
        ]
    },
    "pytorch": {
        "Easy": [
            "What is a Tensor in PyTorch, and how is it different from a NumPy array?",
            "Explain what torch.manual_seed does and why it is important."
        ],
        "Medium": [
            "Explain the difference between Dataset and DataLoader in PyTorch and how they handle batching and shuffling.",
            "What does loss.backward() and optimizer.step() do during the training loop?"
        ],
        "Hard": [
            "Explain how autograd tracks operations and builds the dynamic computation graph in PyTorch.",
            "How would you debug a model that is experiencing vanishing or exploding gradients during training?"
        ]
    },
    "docker": {
        "Easy": [
            "What is the difference between a Docker image and a Docker container?",
            "Explain common Dockerfile directives: FROM, RUN, CMD, and COPY."
        ],
        "Medium": [
            "How do Docker volumes work, and what is the difference between a bind mount and a named volume?",
            "Explain how multi-stage Docker builds optimize image sizes."
        ],
        "Hard": [
            "How does Docker networking work, and what are the differences between bridge, host, and overlay networks?",
            "What are security best practices for writing Dockerfiles and running containers in production?"
        ]
    },
    "kubernetes": {
        "Easy": [
            "What is a Pod in Kubernetes, and why is it considered the smallest deployable unit?",
            "Explain the purpose of a Kubernetes Service."
        ],
        "Medium": [
            "Explain the difference between a Deployment and a StatefulSet in Kubernetes.",
            "What are Liveness and Readiness probes, and why are they critical for self-healing deployments?"
        ],
        "Hard": [
            "Describe the Kubernetes control plane architecture and how components communicate.",
            "How do you implement horizontal pod autoscaling (HPA) based on custom metrics in Kubernetes?"
        ]
    },
    "aws": {
        "Easy": [
            "What is the difference between Amazon S3 and EBS?",
            "Explain the purpose of AWS IAM (Identity and Access Management)."
        ],
        "Medium": [
            "What is a VPC (Virtual Private Cloud), and what are the differences between public and private subnets?",
            "Explain the difference between AWS Lambda (Serverless) and Amazon EC2 instances."
        ],
        "Hard": [
            "How would you design a highly-available, fault-tolerant three-tier architecture in AWS?",
            "Explain AWS IAM Roles vs. IAM Users, and how to implement the principle of least privilege."
        ]
    },
    "react": {
        "Easy": [
            "What is the Virtual DOM in React, and how does it improve rendering performance?",
            "Explain the difference between props and state in React."
        ],
        "Medium": [
            "What are React Hooks? Explain the usage of useEffect and when it triggers re-renders.",
            "Explain the difference between server-side rendering (SSR) and client-side rendering (CSR)."
        ],
        "Hard": [
            "How does React's reconciliation algorithm work (diffing process)?",
            "How would you optimize performance in a React application with a very large, dynamic list of components?"
        ]
    }
}

# Fallback Generic Questions if candidate's skills aren't in the bank
GENERIC_QUESTIONS: Dict[str, List[str]] = {
    "Easy": [
        "Walk us through a technical challenge you solved recently. What was your approach?",
        "How do you approach learning a new technology or programming language?"
    ],
    "Medium": [
        "Describe your preferred development workflow. What tools do you use for version control, testing, and CI/CD?",
        "How do you ensure code quality and maintainability in a team environment?"
    ],
    "Hard": [
        "Explain how you would design a scalable backend API for an application with high traffic spikes.",
        "How do you balance technical debt against the need to deliver features quickly?"
    ]
}

def generate_interview_questions(
    candidate_row: pd.Series,
    matched_skills: List[str],
    missing_skills: List[str]
) -> Dict[str, List[str]]:
    """
    Generates targeted Easy, Medium, and Hard interview questions based on
    the candidate's listed skills, project text, and identified skill gaps.
    """
    cand_name = candidate_row.get("Candidate_Name", "the candidate")
    
    questions = {
        "Easy": [],
        "Medium": [],
        "Hard": []
    }

    # 1. Target Skill Gaps (Missing Skills) - Easy/Medium focus to test core concepts
    gap_skills = [s.strip().lower() for s in missing_skills]
    for skill in gap_skills:
        if skill in QUESTION_BANK:
            # Pick an Easy question from the gap skill
            questions["Easy"].append(f"[Skill Gap: {skill.upper()}] {QUESTION_BANK[skill]['Easy'][0]}")
            # Pick a Medium question from the gap skill
            questions["Medium"].append(f"[Skill Gap: {skill.upper()}] {QUESTION_BANK[skill]['Medium'][0]}")

    # 2. Target Matched Skills - Medium/Hard focus to verify competence
    known_skills = [s.strip().lower() for s in matched_skills]
    for skill in known_skills:
        if skill in QUESTION_BANK:
            # Add Hard question for matched skill
            questions["Hard"].append(f"[Core Skill: {skill.upper()}] {QUESTION_BANK[skill]['Hard'][0]}")
            # Add Medium question if space permits
            if len(questions["Medium"]) < 3:
                questions["Medium"].append(f"[Core Skill: {skill.upper()}] {QUESTION_BANK[skill]['Medium'][1]}")

    # 3. Add Project-Specific Question (Medium/Hard)
    projects_text = candidate_row.get("Projects", "")
    if not pd.isna(projects_text) and str(projects_text).strip() != "":
        proj_clean = str(projects_text).strip()
        questions["Medium"].append(
            f"[Project Context] In your profile, you mentioned: '{proj_clean[:60]}...'. "
            f"What were the core technical constraints and architectural trade-offs you faced in this project?"
        )
    else:
        # Generic project fallback
        questions["Medium"].append(
            "[Project Context] Describe a complex software project you built. "
            "How did you make design decisions regarding language, framework, and data storage?"
        )

    # 4. Experience-Based Question (Hard)
    try:
        cand_exp = float(candidate_row.get("Experience_Years", 0.0))
    except ValueError:
        cand_exp = 0.0
        
    exp_details = candidate_row.get("Experience_Details", "")
    if cand_exp > 5.0 and not pd.isna(exp_details) and str(exp_details).strip():
        questions["Hard"].append(
            f"[Senior Leadership] Given your {cand_exp:.1f} years of experience, how would you handle "
            f"a scenario where a critical production system faces performance bottlenecks due to database connection pooling?"
        )
    elif cand_exp > 0.0:
        questions["Hard"].append(
            f"[Systems Engineering] Reflecting on your {cand_exp:.1f} years of experience: "
            f"Explain how you design applications to handle failure gracefully (e.g. circuit breakers, retries, fallbacks)."
        )

    # 5. Populate and balance using general questions if lists are thin
    for level in ["Easy", "Medium", "Hard"]:
        # If we have no questions for this category, populate from fallback
        if not questions[level]:
            questions[level].extend(GENERIC_QUESTIONS[level])
        # Cap questions per level to a clean number (e.g. 3 questions max per level)
        # Ensure we have at least 2 questions
        if len(questions[level]) < 2:
            questions[level].append(GENERIC_QUESTIONS[level][0])
            
        questions[level] = list(set(questions[level]))[:3] # Deduplicate and slice top 3

    logger.info(f"Generated {sum(len(q) for q in questions.values())} interview questions for {cand_name}")
    return questions

if __name__ == "__main__":
    # Test execution
    test_cand = pd.Series({
        "Candidate_Name": "Alex Rivera",
        "Skills": "Python, React, AWS",
        "Projects": "Created a real-time analytics dashboard with React.",
        "Experience_Years": 4.0,
        "Experience_Details": "Worked as a Full Stack Engineer at WebLabs for 3 years."
    })
    
    matched = ["python", "react", "aws"]
    missing = ["docker", "kubernetes"]
    
    questions = generate_interview_questions(test_cand, matched, missing)
    print("--- Generated Interview Questions ---")
    for lvl, q_list in questions.items():
        print(f"\n[{lvl} Questions]")
        for i, q in enumerate(q_list, 1):
            print(f"  {i}. {q}")
