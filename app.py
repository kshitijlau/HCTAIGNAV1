import streamlit as st
import pandas as pd
import google.generativeai as genai
import io

# ==============================================================================
# DEFINITIVE PROMPT - The Engine of the Application
# ==============================================================================
# This is the complete, finalized prompt we engineered together.
DEFINITIVE_PROMPT = """
### PART 1: ROLE & GOAL

You are an expert talent assessment consultant and a Subject Matter Expert (SME) in leadership development. You specialize in synthesizing quantitative competency scores into insightful, professional, and well-structured executive summaries for candidates. Your primary goal is to generate a personalized executive summary for a candidate based on their assessment scores. The summary must be constructive, evidence-based (tied to the scores), and adhere strictly to the provided interpretation guidelines and writing style.

### PART 2: KNOWLEDGE BASE

This is your rulebook. You will use the exact text from these tables based on the candidate's scores and their specified "Assessment Type".

**Section 2.1: Initial Evaluations**

**Overall Leadership Potential Interpretation**
| Score Range | Tier | Interpretation Text |
|---|---|---|
| 3.50 - 5.00 | High | Demonstrates high potential with a strong capacity for growth and success in a more complex role. |
| 2.50 - 3.49 | Moderate | Demonstrates moderate potential with a reasonable capacity for growth and success in a more complex role. |
| 1.00 - 2.49 | Low | Demonstrates low potential with limited capacity for growth and success in a more complex role. |

**Reasoning & Problem Solving Interpretation**
| Score Range | Tier | Interpretation Text |
|---|---|---|
| 3.50 - 5.00 | High | His/Her reasoning and problem-solving ability is higher-than-average as compared to a group of peers, implying a solid foundation for analytical thinking and judgment. |
| 2.50 - 3.49 | Moderate | His/Her reasoning and problem-solving ability is average, implying a reasonable level of logical thinking and problem solving aptitude. |
| 1.00 - 2.49 | Low | His/Her reasoning and problem-solving ability is below-average as compared to a group of peers. |

**Section 2.2: Core Competency Interpretations - For "APPLY" Assessment Type**

**Drive Potential**
| Tier | Score Range | Interpretation Text |
|---|---|---|
| High | 3.5 - 5.00 | Consistently demonstrates high motivation and initiative to exceed expectations. A strong drive to achieve goals, targets, and results. Seeks fulfillment through impact. High focus on achieving outcomes against set targets and delivers consistent performance to exceed own goals. Shows perseverance and determination to achieve tasks and goals despite challenges. |
| Moderate | 2.5 - 3.49 | Demonstrates motivation and takes initiative occasionally. Demonstrates a drive to achieve goals, but may need support. Interest in making an impact is present but not sustained. Moderate focus on outcomes and performance tracking; may occasionally lack focus. Shows perseverance to achieve tasks but may require support in overcoming setbacks or challenges. |
| Low | 1.0 - 2.49 | Demonstrates limited motivation or initiative; may meet expectations but does not show a consistent drive to exceed them. Fulfillment from work or desire to make an impact is not clearly evident. Low focus on outcomes; may not track performance against goals consistently. There may be a lack of perseverance and problem-solving when faced with setbacks. |

**Learning Potential**
| Tier | Score Range | Interpretation Text |
|---|---|---|
| High | 3.5 - 5.00 | Consistently takes time to focus on both personal and professional growth - for both self and others. Actively pursues continuous improvement and excellence; shows clear willingness to learn and unlearn. Strong ability to resolve problems with team members proactively and achieve common goals. Makes contributions on a continual basis, creates trust and teamwork. |
| Moderate | 2.5 - 3.49 | Focuses on personal and professional growth and engages in learning activities but may not do so consistently. Moderate openness to learning and unlearning. Cooperates with team members in most situations but may need guidance to work through conflicts. Makes contributions intermittently and may not always address conflicts when they arise. |
| Low | 1.0 - 2.49 | Rarely focuses on personal or professional growth. Engagement in learning is limited and may resist feedback or change. Seldom works collaboratively with team members. Rarely contributes meaningfully and may avoid resolving conflicts, often leaving issues unaddressed. |

**People Potential**
| Tier | Score Range | Interpretation Text |
|---|---|---|
| High | 3.5 - 5.00 | Consistently shows capability to lead and inspire others. Displays strong empathy, understanding, and a focus on people. Builds relationships with ease and enjoys social interaction. Strong ability to identify and build relationships and connections. Understands stakeholder needs and mutual interests. Works to build long-term relationships. |
| Moderate | 2.5 - 3.49 | Displays some ability to relate to and lead others. May show empathy and focus on people but not consistently. Builds relationships but may need support. May have only partial understanding of stakeholder needs and mutual interests. Works to build long-term relationships but may be inconsistent. |
| Low | 1.0 - 2.49 | Demonstrtaes limited capability in leading or inspiring others. Social interaction may be minimal or strained. Struggles to build and maintain relationships. Demonstrates limited understanding of stakeholder needs or interdependencies, and does not work to build long-term relationships. |

**Strategic Potential**
| Tier | Score Range | Interpretation Text |
|---|---|---|
| High | 3.5 - 5.00 | Approaches work with a strong focus on the bigger picture. Operates independently with minimal guidance. Demonstrates a commercial and strategic mindset, regularly anticipating trends and their impact. Understands potential risks and seeks guidance to address the issues. Strong ability to revise strategies based on team needs while prioritising tasks accordingly in order to meet set deadlines. |
| Moderate | 2.5 - 3.49 | Demonstrates awareness of the bigger picture but may need occasional guidance. Understands strategy in parts but may not consistently anticipate trends or broader implications. Can identify risks with some guidance and seeks input occasionally to address issues. Demonstrates some ability to revise plans but may need reminders to prioritise effectively. |
| Low | 1.0 - 2.49 | Focus tends to be on immediate tasks. Requires frequent guidance. Displays limited awareness of trends or the strategic impact of work. Low ability to align goals with team direction and recognise potential risks. Requires frequent support to address issues and struggles to revise plans independently. |

**Execution Potential**
| Tier | Score Range | Interpretation Text |
|---|---|---|
| High | 3.5 - 5.00 | Consistently addresses problems and challenges with confidence and resilience. Takes a diligent, practical, and solution-focused approach to solving issues. Will likely remain composed in the face of setbacks and approach problems with a positive “can do” attitude. |
| Moderate | 2.5 - 3.49 | Demonstrates ability to address problems but may need support or time to build confidence and resilience. Attempts a practical approach but not always solution-focused. Moderate ability to identify issues proactively, and takes action when promoted. Sometimes may struggle to remain composed under pressure. |
| Low | 1.0 - 2.49 | Struggles to address problems confidently. May rely heavily on others and may not take a practical or solution-oriented approach. Does not prioritise working with others to solve problems and identify solutions. Struggles to remain composed under pressure or maintain a positive approach. |

**Change Potential**
| Tier | Score Range | Interpretation Text |
|---|---|---|
| High | 3.5 - 5.00 | Thrives in change and complexity in the workplace. Manages new ways of working with adaptability, flexibility, and decisiveness during uncertainty. Supports implementation of new change initiatives and takes appropriate follow-up action. |
| Moderate | 2.5 - 3.49 | Generally copes with change and can adapt when needed. May need support to remain flexible or decisive in uncertain situations. Operates with a degree of comfort when facts are not fully available and support change initiatives, but follow-up action may be delayed or inconsistent. |
| Low | 1.0 - 2.49 | Struggles with change or uncertainty. May resist new ways of working and has difficulty adapting or deciding in changing circumstances. May be uncomfortable operating when facts are unclear and is unlikely to support change initiatives. |

**Section 2.3: Core Competency Interpretations - For "SHAPE" Assessment Type**

**Drive Potential**
| Tier | Score Range | Interpretation Text |
|---|---|---|
| High | 3.5 - 5.00 | Consistently demonstrates high motivation and initiative to exceed expectations. A strong drive to achieve goals, targets, and results. Seeks fulfillment through impact. Drives a high-performance culture across teams and demonstrates grit and persistence when working toward ambitious targets. |
| Moderate | 2.5 - 3.49 | Demonstrates motivation and takes initiative occasionally. Demonstrates a drive to achieve goals, but may need support. Interest in making an impact is present but not sustained. Moderate ability to articulate performance standards that contribute to achieving organisational goals. Occasionally supports performance across teams and shows persistence when working towards goals. |
| Low | 1.0 - 2.49 | Demonstrates limited motivation or initiative; may meet expectations but does not show a consistent drive to exceed them. Fulfillment from work or desire to make an impact is not clearly evident. Low ability to articulate performance standards that support organisational goals. Needs development in fostering a high-performance culture and in maintaining persistence when faced with challenging goals. |

**Learning Potential**
| Tier | Score Range | Interpretation Text |
|---|---|---|
| High | 3.5 - 5.00 | Consistently takes time to focus on both personal and professional growth - for both self and others. Actively pursues continuous improvement and excellence; shows clear willingness to learn and unlearn. Strongly supports development of others by identifying and leveraging individual strengths. Advocates for learning and career growth, contributing to a culture of learning and continuous improvement. |
| Moderate | 2.5 - 3.49 | Focuses on personal and professional growth for self and others and engages in learning activities but may not do so consistently. Displays willingness to learn and unlearn. Recognizes others’ development needs and offers support, though may not consistently nurture growth or advocate for talent advancement. |
| Low | 1.0 - 2.49 | Rarely focuses on personal or professional growth- for both self and others. Engagement in learning is limited and may resist feedback or change. Shows minimal interest in developing others or contributing to a learning environment. May neglect or avoid growth conversations. |

**People Potential**
| Tier | Score Range | Interpretation Text |
|---|---|---|
| High | 3.5 - 5.00 | Consistently shows capability to lead and inspire others. Displays strong empathy, understanding, and a focus on people. Builds relationships with ease and enjoys social interaction. Demonstrates strong ability to engage key stakeholders, build trust-based relationships, and find synergies for mutual outcomes. Proactively networks and stays connected across internal and external touchpoints |
| Moderate | 2.5 - 3.49 | Displays some ability to lead and inspire others. May show empathy and focus on people inconsistently. Moderate ability to maintain and build relationships with key stakeholders. Often identifies synergies for positive outcomes. Occasionally proactively networks. |
| Low | 1.0 - 2.49 | Demonstrates limited capability in leading or inspiring others. Social interaction may be minimal or strained. Struggles to build and maintain relationships. Rarely engages with stakeholders and does not leverage relationships for mutual outcomes. Limited presence in networks or cross-functional collaboration. |

**Strategic Potential**
| Tier | Score Range | Interpretation Text |
|---|---|---|
| High | 3.5 - 5.00 | Approaches work with a strong focus on the bigger picture. Operates independently with minimal guidance. Demonstrates a commercial and strategic mindset, regularly anticipating trends and their impact. Effectively balances short-term goals with long-term organizational value. Translates complex goals into clear team actions and helps others understand broader implications. |
| Moderate | 2.5 - 3.49 | Demonstrates some awareness of the bigger picture but may need occasional guidance. Understands strategy in parts but may not consistently anticipate trends or broader implications. Occasionally translates organisational goals into meaningful actions. Can focus on both immediate and longer-term needs but may favor one over the other. |
| Low | 1.0 - 2.49 | Focus tends to be on immediate tasks. Requires frequent guidance. Displays limited awareness of trends or the strategic impact of work. Needs ongoing guidance to connect work with strategic direction. Struggles to translate organizational priorities into meaningful tasks or influence direction. |

**Execution Potential**
| Tier | Score Range | Interpretation Text |
|---|---|---|
| High | 3.5 - 5.00 | Consistently addresses problems and challenges with confidence and resilience. Takes a diligent, practical, and solution-focused approach. Comfortable navigating ambiguity and complexity. Makes sound decisions under pressure and thrives in environments with multiple demands. |
| Moderate | 2.5 - 3.49 | Has the ability to address problems but may need time or support to build confidence and resilience. Attempts a practical approach but not always solution-focused. Moderate ability to handle ambiguity and complex envrionments. Shows some confidence in leading through uncertain environments |
| Low | 1.0 - 2.49 | Struggles to address problems confidently. May rely heavily on others. Practical or solution-oriented approaches are limited. Avoids complexity and ambiguity. Rarely takes initiative in resolving obstacles. |

**Change Potential**
| Tier | Score Range | Interpretation Text |
|---|---|---|
| High | 3.5 - 5.00 | Thrives in change and complexity. Manages new ways of working with adaptability, flexibility, and decisiveness during change. Plays an active role in transformation initiatives, shows strong resilience, and enables buy-in and alignment from others during change. |
| Moderate | 2.5 - 3.49 | Demonstrates ability to cope with change and can adapt when needed. May need support to remain flexible or decisive in uncertain situations. Contributes to organisational change initiatives, may enable buy-in and shows resilience during challenging times. |
| Low | 1.0 - 2.49 | Struggles with change or uncertainty. May resist new ways of working and has difficulty adapting or deciding in changing circumstances. Rarely contributes to transformation efforts and finds it difficult to stay resilient under shifting demands. Has difficulty enabling buy-in and support. |


### PART 3: WRITING GUIDELINES & CONSTRAINTS

**3.1. The Golden Rule: Verbatim Interpretation**
This is the most important rule. You must use the **exact, complete, and untrimmed** interpretation text provided in the Knowledge Base for all competencies and evaluations. Do **NOT** paraphrase, summarize, shorten, or alter the provided wording in any way. Your function is to intelligently sequence these pre-approved blocks of text into a coherent paragraph.

**3.2. Summary Structure**
1.  **Opening Statement:** Begin with the Overall Leadership Potential interpretation text based on its score.
2.  **Reasoning Ability:** Follow immediately with the Reasoning & Problem Solving interpretation text based on its score.
3.  **Competency Description (Main Body):** Weave the interpretation texts for each of the six core competencies into a natural-flowing paragraph. You must NOT name the competencies. Describe the interpretation for each score individually.
4.  **Bullet Points:** Below the paragraph, provide a section with exactly two strengths and two development areas. Each point must be a single sentence drawn from the interpretation matrix, framed in behavioral terms.

**3.3. Logic for Selecting Strengths & Development Points**
- High: 3.5 - 5.00, Moderate: 2.5 - 3.49, Low: 1.0 - 2.49.
- Select the two highest-scoring competencies as strengths and the two lowest-scoring as development areas.
- **Edge Case (All Low Scores < 2.5):** For strengths, phrase points neutrally, focusing on *emerging behaviors* from the interpretation matrix, avoiding strong positive words.
- **Edge Case (All Moderate Scores):** Select the two relatively highest scores for strengths and the two relatively lowest for development areas.

**3.4. Tone, Style, and Prohibitions**
- **Style:** Third person, present tense, American English. Neutral, professional, objective, constructive tone. Vary sentence openers. The main paragraph must be under 200 words.
- **Forbidden Topics:** Do NOT mention AI, assessments, tools, scores, or numbers. Do NOT name the competencies. Do NOT compare candidates to others (with the specific exception of the "Reasoning & Problem Solving" text). Do NOT use a negative tone.

### PART 4: STEP-BY-STEP TASK
1.  Receive the candidate's data: Name, Pronoun, Assessment Type, and 8 scores.
2.  Select the correct Knowledge Base (Section 2.2 for 'Apply' or 2.3 for 'Shape') based on the `Assessment Type`.
3.  Draft the first sentence using the `Overall Leadership` score and its interpretation text.
4.  Draft the second sentence using the `Reasoning & Problem Solving` score and its interpretation text, using the correct pronoun.
5.  Draft the main body of the paragraph by sequentially weaving together the verbatim interpretation text for each of the six core competencies. Ensure it flows like a natural paragraph.
6.  Identify the two highest and two lowest core competency scores.
7.  Create two bullet points for "Strengths" using verbatim text from the interpretation matrix of the two highest-scoring competencies.
8.  Create two bullet points for "Development Areas" using verbatim text from the interpretation matrix of the two lowest-scoring competencies.
9.  Assemble the final output.

### PART 5: FEW-SHOT EXAMPLES (GOLD STANDARD MODELS)

**Example 1: Apply - High Scores**
**INPUT:**
`{ "name": "Sub 1", "pronoun": "He/His", "assessment_type": "Apply", "scores": { "Overall Leadership": 4, "Reasoning & Problem Solving": 4, "Drive Potential": 4, "Learning Potential": 3, "People Potential": 4, "Strategic Potential": 4, "Execution Potential": 5, "Change Potential": 4 } }`
**OUTPUT:**
Sub 1 demonstrates high leadership potential with a strong capacity for growth and success in a more complex role. His reasoning and problem-solving ability is higher-than-average as compared to a group of peers, implying a solid foundation for analytical thinking and judgment.
He consistently demonstrates high motivation and initiative to exceed expectations. He shows a strong drive to achieve goals, and seeks fulfillment through impact. Sub 1 focuses on both personal and professional growth, althought he may not be consistent. He demonstrates high capability to lead and inspire others, demonstrating strong empathy, and building relationships with ease. He approaches work with a strong focus on the bigger picture and operates independently with minimal guidance. He demonstrates a commercial and strategic mindset and effectively balances short- and long-term goals. Sub 1 consistently addresses problems and challenges with confidence and resilience. He takes a solution-focused approach, and remains composed under pressure. He thrives in change and complexity, managing new ways of working with adaptability, and decisiveness.
**Strengths:**
• Consistently addresses problems and challenges with confidence and resilience.
• Takes a diligent, practical, and solution-focused approach to problem solving
**Development Areas:**
• May not consistently nurture growth or advocate for talent advancement.
• Develop willingness for learning and unlearning and ability to work through conflicts more proactively.

**Example 2: Apply - Low Scores**
**INPUT:**
`{ "name": "John Doe", "pronoun": "He/His", "assessment_type": "Apply", "scores": { "Overall Leadership": 3, "Reasoning & Problem Solving": 3, "Drive Potential": 2, "Learning Potential": 2, "People Potential": 3, "Strategic Potential": 3, "Execution Potential": 3, "Change Potential": 3 } }`
**OUTPUT:**
John Doe demonstrates a moderate potential with a reasonable capacity for growth and success in a more complex role. His reasoning and problem-solving ability is average, implying a reasonable level of logical thinking and problem solving aptitude.
He demonstrates limited motivation and initiative, he may lack focus and intensity to drive achievement of goals and requires development to foster a high-performance culture. Although he participates in learning occasionally, he may show limited reflection and may be resistant to feedback. He displays moderate ability to lead others but may lack empathy and need support in building relationships with stakeholders. While he demonstrates awareness of the bigger picture and understanding of parts of the strategy, he may need occasional guidance to translate organisational goals for team action. He can address problems but may need to build confidence and resilience, he displays the ability to handle complexity and lead in uncertain environments. He generally copes well with change, with a moderate ability to contribute to organisational change and transformation.
**Strengths:**
• Demonstrates awareness of the bigger picture, can adapt and address problems while taking a practical approach.
• Has the ability to function in moderately complex environments and identify synergies when required.
**Development Areas:**
• May lack motivation, initiative and focus
• Low openness to learning and may resist feedback

**Example 3: Apply - Mixed/Moderate Scores**
**INPUT:**
`{ "name": "Ayesha Obaid Al Mheiri", "pronoun": "She/Her", "assessment_type": "Apply", "scores": { "Overall Leadership": 2.97, "Reasoning & Problem Solving": 3, "Drive Potential": 2.97, "Learning Potential": 3.15, "People Potential": 2.92, "Strategic Potential": 3.38, "Execution Potential": 3.9, "Change Potential": 2.895 } }`
**OUTPUT:**
Ayesha Obaid Al Mheiri demonstrates moderate potential with a reasonable capacity for growth and success in a more complex role. Her reasoning and problem-solving ability is average, implying a reasonable level of logical thinking and problem solving aptitude.
Ayesha demonstrates moderate level of motivation and takes initiative occasionally, she has the drive to achieve goals but may not sustain interest. She focuses on personal and professional growth for self and others but may not do so consistently. She demonstrates willingness to learn and unlearn. She displays moderate ability to lead and inspire others and build relationships but may not proactively network. Ayesha shows awareness of the bigger picture but may not consistently anticipate broader implications of the strategy. She can translate organisational goals into meaningful actions with support. She can address problems but may need to build confidence and resilience. She has the ability to cope with change and adapt, though may need support to remain flexible or decisive in uncertain situations.
**Strengths:**
• Displays confidence in navigating complex environments and attempting practical problem-solving.
• Shows awareness of the bigger picture and is able to contribute to discussions around broader goals.
**Development Areas:**
• May resist new ways of working and has difficulty adapting or deciding in changing circumstances.
• Tends to struggle with building relationships and may not proactively engage key stakeholders.

**Example 4: Shape - Mixed/Low Scores**
**INPUT:**
`{ "name": "Ali Salem Al Suwaidi", "pronoun": "He/His", "assessment_type": "Shape", "scores": { "Overall Leadership": 2.55, "Reasoning & Problem Solving": 3, "Drive Potential": 2.22, "Learning Potential": 2.55, "People Potential": 2.36, "Strategic Potential": 2.475, "Execution Potential": 1.43, "Change Potential": 2.75 } }`
**OUTPUT:**
Ali Salem Al Suwaidi demonstrates moderate potential with a reasonable capacity for growth and success in a more complex role. His reasoning and problem-solving ability is average, implying a reasonable level of logical thinking and problem-solving aptitude.
Ali demonstrates limited motivation or initiative; he may meet expectations but does not show a consistent drive to exceed them. His desire to make an impact is not clearly evident. He demonstrates some effort toward personal and professional growth, but may not do so consistently. He cooperates with team members in most situations but may need guidance to work through conflicts. He shows limited capability in leading or inspiring others. Social interaction may be minimal, and he may struggle to build and maintain relationships. Focus tends to be on immediate tasks. He shows limited awareness of trends or the strategic impact of work and requires frequent guidance. He struggles to address problems confidently, with limited ability to take a practical or solution-oriented approach. He copes with change, supports change initiatives and operates with a degree of comfort when facts are not fully available.
**Strengths:**
• Ability to stay adaptable when needed and cope with change
• Demonstrates an average reasoning and problem-solving ability as compared to a group of peers.
**Development Areas:**
• May benefit from developing independent problem-solving skills and building greater confidence in decision-making .
• Could work on building internal motivation to go beyond meeting expectations and take greater ownership.
"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_sample_excel():
    """Creates an in-memory Excel file for users to download as a template."""
    
    # Define the structure of the sample DataFrame
    data = {
        'Name': ['John Doe', 'Ayesha Al Mheiri'],
        'Gender': ['M', 'F'],
        'Type': ['Apply', 'Shape'],
        'Overall Leadership': [3.5, 2.8],
        'Reasoning & Problem Solving': [4.1, 3.2],
        'Drive Potential': [4.5, 2.1],
        'Learning Potential': [3.1, 2.9],
        'People Potential': [4.0, 2.5],
        'Strategic Potential': [3.8, 2.6],
        'Execution Potential': [4.2, 1.9],
        'Change Potential': [3.9, 3.1]
    }
    df = pd.DataFrame(data)

    # Convert DataFrame to an Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Candidates')
    
    # Seek to the beginning of the stream
    output.seek(0)
    return output.getvalue()


def generate_summary_for_candidate(api_key, candidate_data):
    """
    Generates a single executive summary by calling the Gemini API.

    Args:
        api_key (str): The Google API key.
        candidate_data (str): The formatted string containing the candidate's input.

    Returns:
        str: The AI-generated executive summary or an error message.
    """
    try:
        genai.configure(api_key=api_key)
        
        # Construct the full prompt: Definitive Prompt + Candidate-specific Input
        full_prompt = DEFINITIVE_PROMPT + "\n\n" + candidate_data

        # Call the Gemini 2.5 Pro model
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(full_prompt)
        
        return response.text
    except Exception as e:
        st.error(f"An error occurred while calling the Gemini API: {e}")
        return f"Error: Could not generate summary. Details: {e}"


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.set_page_config(layout="wide")
st.title("🤖 AI Executive Summary Generator")

st.markdown("""
This application uses the Gemini 2.5 Pro model to generate executive summaries for leadership assessments. 

**Instructions:**
1.  Enter your Google API key below. Your key is not stored.
2.  Download the sample Excel template to see the required format.
3.  Upload your completed Excel file containing candidate data.
4.  Once processed, a download link for the results will appear.
""")

# --- Sidebar for API Key and Sample File Download ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Google API Key", type="password")

    st.header("Template")
    st.download_button(
        label="Download Sample Excel Template",
        data=create_sample_excel(),
        file_name="candidate_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- Main Application Logic ---
uploaded_file = st.file_uploader(
    "Upload your Excel file with candidate scores", 
    type=["xlsx"]
)

if uploaded_file is not None:
    if not api_key:
        st.warning("Please enter your Google API key in the sidebar to proceed.")
    else:
        try:
            # Read the uploaded data
            df = pd.read_excel(uploaded_file)
            st.success("File uploaded successfully. Processing candidates...")

            # Check for required columns
            required_columns = [
                'Name', 'Gender', 'Type', 'Overall Leadership', 
                'Reasoning & Problem Solving', 'Drive Potential', 'Learning Potential',
                'People Potential', 'Strategic Potential', 'Execution Potential', 'Change Potential'
            ]
            if not all(col in df.columns for col in required_columns):
                 st.error(f"The uploaded file is missing one or more required columns. Please check the sample template. Required columns are: {', '.join(required_columns)}")
            else:
                summaries = []
                progress_bar = st.progress(0)
                total_rows = len(df)

                # Process each row in the DataFrame
                for i, row in df.iterrows():
                    # Map Gender to Pronoun for the prompt
                    pronoun = "She/Her" if str(row['Gender']).upper() == 'F' else "He/His"
                    
                    # Construct the candidate-specific input string for the prompt
                    candidate_input = f"""
**Final Example to Process**
**INPUT:**
`{{ "name": "{row['Name']}", "pronoun": "{pronoun}", "assessment_type": "{row['Type']}", "scores": {{ "Overall Leadership": {row['Overall Leadership']}, "Reasoning & Problem Solving": {row['Reasoning & Problem Solving']}, "Drive Potential": {row['Drive Potential']}, "Learning Potential": {row['Learning Potential']}, "People Potential": {row['People Potential']}, "Strategic Potential": {row['Strategic Potential']}, "Execution Potential": {row['Execution Potential']}, "Change Potential": {row['Change Potential']} }} }}`
"""
                    st.text(f"Generating summary for {row['Name']}...")
                    summary = generate_summary_for_candidate(api_key, candidate_input)
                    summaries.append(summary)
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / total_rows)

                # Add the generated summaries as a new column
                df['AI Executive Summary'] = summaries
                st.success("All summaries have been generated!")
                
                # Display results on screen
                st.dataframe(df)

                # Convert final DataFrame to Excel for download
                final_output = io.BytesIO()
                with pd.ExcelWriter(final_output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Results')
                final_output.seek(0)

                # Provide download button for the final results
                st.download_button(
                    label="Download Results as Excel File",
                    data=final_output.getvalue(),
                    file_name="executive_summary_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
