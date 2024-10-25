import json
import logging
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from config.settings import OPENAI_API_KEY
from src.utils import info_extraction

# Base class for agents
class WrapperFrame_v1:
    def __init__(self, client):
        self.client = client

# Narrative Agents
# 1. Positive Sentiment Narratives
class ES_Agent_Positive(WrapperFrame_v1):
    def __init__(self, client):
        super().__init__(client)
        self.ES_agent = """
    Your task is to analyze and categorize a social media post {input_text} from El salvador according to predefined narrative themes that reflect positive sentiment with validated context.


    Narratives to Detect:

      ESP1: The Government’s Firm Approach Restoring Law and Order.
      Example: "The government's hardline security policies have restored law and order in El Salvador, reducing crime rates and bringing a sense of safety to local communities."
      Keywords: law and order, security, crime reduction, safety, firm approach.

      ESP2: El Salvador is Becoming a Model of Security and Strong Leadership in Fighting Gang Violence.
          Example: "El Salvador's leadership in combating gang violence is becoming a model for other countries in the region, showcasing the government's resolve and effectiveness."
          Keywords: security model, leadership, gang violence, regional example, safety.

      ESP3: Government Policies Attracting Support and International Investment.
          Example: "International companies are flocking to El Salvador, drawn by favorable government policies that support innovation and economic growth."
          Keywords: support, international investment, government policies, economic growth, innovation.

      ESP4: El Salvador at the Forefront of Economic Modernization and Cryptocurrency Adoption.
          Example: "El Salvador’s adoption of Bitcoin is positioning the country as a pioneer in financial innovation, paving the way for economic modernization."
          Keywords: cryptocurrency, Bitcoin, economic modernization, innovation, financial pioneer.

      ESP5: Bitcoin Policy Opening Doors for Innovation and Attracting Entrepreneurs and International Investors.
          Example: "El Salvador’s Bitcoin policy is attracting a new wave of entrepreneurs and international investors, driving technological innovation and boosting the economy."
          Keywords: Bitcoin policy, innovation, entrepreneurship, international investors, economic boost.

      ESP6: The Bitcoin Revolution Could Break the Country’s Dependence on Traditional Financial Systems.
          Example: "By embracing Bitcoin, El Salvador is reducing its reliance on traditional financial systems and promoting financial independence."
          Keywords: Bitcoin revolution, financial independence, traditional systems, innovation.

      ESP7: El Salvador Affirms Sovereignty and Emerges as a Regional Leader by Standing up to Foreign Pressure.
          Example: "El Salvador is asserting its sovereignty, showing resilience in the face of foreign pressure, and positioning itself as a leader in Central America."
          Keywords: sovereignty, leadership, resilience, foreign pressure, regional influence.

      ESP8: Independent Policies Lead to a New Era of Self-Sufficiency and Greater Regional Influence.
          Example: "The government's independent policies are fostering a new era of self-sufficiency, boosting El Salvador's regional influence."
          Keywords: independent policies, self-sufficiency, regional influence, leadership, progress.

      ESP9: Alternative Media Providing Truth Suppressed by Traditional Media.
          Example: "Alternative media outlets in El Salvador are giving voice to suppressed truths, countering narratives controlled by traditional media."
          Keywords: alternative media, suppressed truth, transparency, traditional media.

      ESP10: Government Promoting Transparency Through its Own Media Channels.
          Example: "The government’s media channels are promoting transparency by delivering unfiltered information directly to the public."
          Keywords: transparency, government media, unfiltered information, public access, direct communication.

      ESP11: Economic Growth Promised by Government Reforms and Innovation.
          Example: "Thanks to recent reforms and a focus on innovation, El Salvador is on the path to sustained economic growth."
          Keywords: economic growth, reforms, innovation, future prosperity, development.

      ESP12: Government Promoting Equality Policies and Empowering Women.
          Example: "New government policies aimed at promoting gender equality are empowering women across different sectors of society."
          Keywords: equality, women’s rights, empowerment, government policies, gender equality.

      ESP13: Support Programs for Vulnerable Women Reinforcing Commitment to Gender Equality.
          Example: "The government’s support programs for women in vulnerable situations are reinforcing its commitment to gender equality and social justice."
          Keywords: support programs, vulnerable women, gender equality, empowerment, social justice.

      ESP14: Policies for LGBT Inclusion and Respect for Rights.
          Example: "El Salvador is leading the way in LGBT inclusion, with new policies that protect and respect the rights of the LGBT community."
          Keywords: LGBT inclusion, rights, respect, protection, government policies.

      ESP15: Government Promoting Tolerance and Combatting Discrimination.
          Example: "Efforts to promote tolerance and fight discrimination are creating a more inclusive society in El Salvador."
          Keywords: tolerance, anti-discrimination, inclusivity, social progress, government efforts.

      ESP16: Strengthening Institutions and Creating Transparency Mechanisms to Combat Corruption.
          Example: "The government has strengthened its institutions and implemented transparency mechanisms to combat corruption, ensuring accountability at all levels."
          Keywords: strengthened institutions, transparency, anti-corruption, accountability, reforms.

      ESP17: Public Support for Anti-Corruption Initiatives Increasing Trust in Institutions.
          Example: "El Salvador's anti-corruption efforts are gaining widespread public support, restoring trust in governmental institutions."
          Keywords: anti-corruption, public support, institutional trust, transparency, government integrity.

    Instructions for Classification:

    1- Read carefully the input {input_text}.
    2- Determine which narrative(s) it supports based on the content and sentiment expressed.
    3- Generate a Json structure:
              (
                "classification": ["ESP1", "ESP2", "ESP3", "ESP4", "ESP5", "ESP6", "ESP7", "ESP8", "ESP9", "ESP10", "ESP11", "ESP12", "ESP13", "ESP14", "ESP15","ESP16", "ESP17"]
                 at most 2 categories,
                "reasoning": reasoning of the answer in maximum 50 words.
              ).
        """
        self.prompt_ES_agent = PromptTemplate(template=self.ES_agent, input_variables=["input_text"])
        self.llm_chain_ES_agent = LLMChain(prompt=self.prompt_ES_agent, llm=self.client)

    def _run_agent(self, input_text):
        try:
            ES_output = self.llm_chain_ES_agent.run({"input_text": input_text})
            return ES_output
        except Exception as e:
            logging.error(e)
            return "Error ES Positive agent"

# 2. Negative Sentiment Narratives
class ES_Agent_Negative(WrapperFrame_v1):
    def __init__(self, client):
        super().__init__(client)
        self.ES_agent = """
    Your task is to analyze and categorize a social media post {input_text} from El salvador according to predefined narrative themes that reflect positive sentiment with validated context.


    Narratives to Detect:

      ESN1: The Government’s Rigid Approach is Compromising Civil Liberties.
      Example: "The government’s strict security policies are being criticized for compromising civil liberties under the guise of law and order."
      Keywords: civil liberties, law and order, government rigidity, criticism, human rights.

      ESN2: Security Policies Lean Towards Authoritarianism, Risking Escalation of Violence.
          Example: "El Salvador’s security measures are showing authoritarian tendencies, increasing concerns that violence could escalate instead of being reduced."
          Keywords: authoritarianism, security policies, violence, civil unrest, escalation.

      ESN3: Government Actions Attracting International Scrutiny, Deterring Investment.
          Example: "El Salvador is facing international scrutiny over its policies, and there are concerns that foreign investment may be deterred."
          Keywords: international scrutiny, government policies, deterred investment, foreign relations, economic risk.

      ESN4: Cryptocurrency Policies are Risky, Endangering Economic Stability.
          Example: "El Salvador’s decision to adopt Bitcoin is seen as risky, potentially threatening the country’s economic stability due to speculative investments."
          Keywords: cryptocurrency, Bitcoin, economic risk, speculative investments, financial instability.

      ESN5: Bitcoin Policy Isolating the Country from Traditional Financial Allies.
          Example: "The Bitcoin policy is creating uncertainty, as El Salvador becomes more isolated from its traditional financial partners."
          Keywords: Bitcoin policy, financial isolation, uncertainty, traditional finance, international business.

      ESN6: Push for Bitcoin Deepens Economic Vulnerability.
          Example: "The push for Bitcoin could deepen El Salvador's economic vulnerability by increasing dependence on volatile digital assets."
          Keywords: Bitcoin, economic vulnerability, dependence, volatility, financial instability.

      ESN7: El Salvador’s Distancing from International Partners is Alienating Allies.
          Example: "El Salvador’s foreign policies are alienating its regional allies, weakening its influence on the international stage."
          Keywords: foreign policy, international relations, alienation, regional allies, diminished influence.

      ESN8: Independent Policies Increase Isolation from Key Support Networks.
          Example: "By pursuing independent policies, El Salvador is detaching itself from vital support networks, increasing its international isolation."
          Keywords: independent policies, isolation, support networks, foreign relations, government strategy.

      ESN9: Alternative Media is Promoting Polarizing Narratives.
          Example: "Some alternative media outlets in El Salvador are fostering polarizing narratives, raising questions about the reliability of the information they provide."
          Keywords: alternative media, polarizing narrative, reliability, media trust, information control.

      ESN10: Government-Controlled Media Reduces Transparency.
          Example: "By controlling its own media channels, the government may be reducing transparency and limiting access to independent sources of information."
          Keywords: government control, media, reduced transparency, limited information, censorship concerns.

      ESN11: Government Reforms are Creating Economic Uncertainty.
          Example: "Recent government reforms have created uncertainty in the market, posing high risks that threaten the country’s long-term stability."
          Keywords: government reforms, economic uncertainty, high risk, market instability, future concerns.

      ESN12: Insufficient Gender Policies Allow Discrimination Against Women to Persist.
          Example: "Despite reforms, violence and discrimination against women continue, showing that the government's gender policies are insufficient."
          Keywords: gender policies, insufficient reform, violence against women, discrimination, government failure.

      ESN13: Lack of Funding Weakens Support Programs for Women’s Rights.
          Example: "Government programs aimed at protecting women’s rights are underfunded, severely limiting their effectiveness."
          Keywords: women’s rights, underfunding, lack of support, program effectiveness, gender equality.

      ESN14: Lack of Legal Protection for the LGBT Community Increases Vulnerability.
          Example: "Without strong legal protections, the LGBT community in El Salvador faces significant vulnerabilities, limiting their full participation in society."
          Keywords: LGBT rights, legal protection, vulnerability, exclusion, government policy.

      ESN15: Government Criticized for Failing to Actively Promote LGBT Rights.
          Example: "Current government policies have been criticized for not doing enough to actively promote the rights of the LGBT community, leading to an atmosphere of exclusion."
          Keywords: LGBT rights, government inaction, exclusion, criticism, equality.

      ESN16: Corruption Allegations Weaken Public Trust.
          Example: "Persistent corruption allegations against government officials are eroding public trust and undermining the effectiveness of institutions."
          Keywords: corruption, public trust, institutional effectiveness, government officials, allegations.

      ESN17: Lack of Transparency Raises Concerns About Government Integrity.
          Example: "The lack of transparency in government processes is raising concerns about the integrity and accountability of public management."
          Keywords: transparency, government integrity, accountability, concerns, public management.

    Instructions for Classification:

    1- Read carefully the input {input_text}.
    2- Determine which narrative(s) it supports based on the content and sentiment expressed.
    3- Generate a Json structure:
              (
                "classification": ["ESN1", "ESN2", "ESN3", "ESN4", "ESN5", "ESN6", "ESN7", "ESN8", "ESN9", "ESN10", "ESN11", "ESN12", "ESN13", "ESN14", "ESN15","ESN16","ESN17"]
                 at most 2 categories,
                "reasoning": reasoning of the answer in maximum 50 words.
              ).
        """
        self.prompt_ES_agent = PromptTemplate(template=self.ES_agent, input_variables=["input_text"])
        self.llm_chain_ES_agent = LLMChain(prompt=self.prompt_ES_agent, llm=self.client)

    def _run_agent(self, input_text):
        try:
            ES_output = self.llm_chain_ES_agent.run({"input_text": input_text})
            return ES_output
        except Exception as e:
            logging.error(e)
            return "Error ES Negative agent"

# Class to identify entities and generate optimized queries for a web search
class ClassAgent:
    def __init__(self, llm_35):
        self.llm_35 = llm_35
        self.class_agent = """
        You are an agent with the task of analyzing a tweet input {input} from El Salvador.
        You are tasked with identifying entities mentioned or referred to in the input and generating an optimized query to launch a web search concerning the input.
        Instructions:
            1- Read the input carefully.
            2- Identify the real entities in it (actors or institutions from El Salvador).
            3- Segment the original input depending on the entity they refer to.
            4- Reframe the tweet for an optimal web search in Spanish.
            5- Generate a Json output following this pattern:
                (
                "entities": [Identify the real entities in it (actors or institutions from El Salvador) in the tweet as a list],
                "segments": [segments of the original input divided by the entity referred to],
                "search": Reframe the tweet for an optimal web search in Spanish
                )
        """
        self.prompt_class_agent = PromptTemplate(template=self.class_agent, input_variables=["input"])
        self.llm_chain_class_agent = LLMChain(prompt=self.prompt_class_agent, llm=self.llm_35)

    def _run_class_branch(self, input):
        try:
            class_agent_output = self.llm_chain_class_agent.run({"input": input})
            return class_agent_output
        except Exception as e:
            logging.error(e)
            return "Error class layer"

# Sentiment analysis agent to determine sentiment of input
class SentimentAgent:
    def __init__(self, llm_35):
        self.llm_35 = llm_35
        self.sentiment_agent = """
        You are an expert agent that has to analyze and classify a tweet {input} from El Salvador according to its sentiment.
        Your job is to categorize each tweet into one of two categories based on its sentiment: 1 (positive) or -1 (negative).
        Guidelines:
        . 1 (Positive): The tweet expresses a positive sentiment, such as happiness, praise, or admiration.
        . -1 (Negative): The tweet expresses a negative sentiment, such as criticism, disappointment, or disapproval.
        Instructions:
            - Read the input carefully.
            - Analyze the context.
            - Determine the sentiment expressed based on the content, considering the context is El Salvador.
            - Generate a Json output following this pattern:
                    (
                      "label": 1 or -1,
                      "reasoning": reasoning of the answer in 50 words.
                    ).
        """
        self.prompt_sentiment_agent = PromptTemplate(template=self.sentiment_agent, input_variables=["input"])
        self.llm_chain_sentiment_agent = LLMChain(prompt=self.prompt_sentiment_agent, llm=self.llm_35)

    def _run_sentiment_branch(self, input):
        try:
            sentiment_agent_output = self.llm_chain_sentiment_agent.run({'input': input})
            return sentiment_agent_output
        except Exception as e:
            logging.error(e)
            return "Error sentiment layer"

# Context builder agent to create context from search results
class ContextBuilderAgent:
    def __init__(self, llm_35):
        self.llm_35 = llm_35
        self.context_builder_agent = """
        You are an agent with the task of creating a context from a social media input {input} from El Salvador.
        You are provided with the result of a web search {search_output}.
        Your job is to :
        1- Review the input {input} and search output {search_output}.
        2- Verify all the entries of the search_output {search_output} one by one and analyze the alignment with the input {input}.
        3- Build a context from the search_output to help understand the context of the input.
        4- Provide a json output:
                (
                "context": Context to help understand the input.
                )
        """
        self.prompt_context_builder_agent = PromptTemplate(template=self.context_builder_agent, input_variables=["input", "search_output"])
        self.llm_chain_context_agent = LLMChain(prompt=self.prompt_context_builder_agent, llm=self.llm_35)

    def _run_context_builder_branch(self, input, search_output):
        try:
            context_agent_builder_output = self.llm_chain_context_agent.run({'input': input, 'search_output': search_output})
            return context_agent_builder_output
        except Exception as e:
            logging.error(e)
            return "Error context building layer"

# Class to define the overall processing pipeline
class VerificationPipeline:
    def __init__(self, llm_35, llm_4, input):
        self.class_agent = ClassAgent(llm_35)
        self.sentiment_agent = SentimentAgent(llm_35)
        self.llm_4 = llm_4
        self.llm_35 = llm_35
        self.input = input
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_news(self):
        self.logger.info('Process Initialized')
        classification_result = self.classify_news(self.input)
        if not classification_result or 'error' in classification_result:
            self.logger.error(f"Classification failed: {classification_result}")
            return classification_result

        entities = classification_result.get("entities", [])
        segments = classification_result.get("segments", [])
        final_output = []

        for entity, segment in zip(entities, segments):
            self.logger.info(f"Processing entity: {entity}")
            sentiment_analysis_result = self.analyze_sentiment(segment)
            if 'error' in sentiment_analysis_result:
                return sentiment_analysis_result

            sentiment_label = sentiment_analysis_result['label']
            sentiment_description = "Positive" if sentiment_label == 1 else "Negative"

            intention_result = self.determine_intention(sentiment_label, segment)
            if 'error' in intention_result:
                return intention_result

            narrative_descriptions = [INTENTION_DICT.get(code, "Unknown Narrative") for code in intention_result['classification']]
            entity_output = {
                "entity": entity,
                "segment": segment,
                "sentiment_label": sentiment_label,
                "sentiment_description": sentiment_description,
                "intention_classification": intention_result['classification'],
                "narrative_descriptions": narrative_descriptions,
                "reasoning": intention_result['reasoning'],
            }
            final_output.append(entity_output)

        return final_output

    def classify_news(self, input):
        result = self.class_agent._run_class_branch(input)
        try:
            data = json.loads(result)
            return data
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding classification result: {result}")
            return {"error": "Failed to classify the news."}

    def analyze_sentiment(self, input):
        result = self.sentiment_agent._run_sentiment_branch(input)
        try:
            data = json.loads(result)
            return data
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding sentiment result: {result}")
            return {"error": "Failed to analyze sentiment."}

    def determine_intention(self, sentiment_label, input_text):
        if sentiment_label == 1:
            agent = ES_Agent_Positive(self.llm_35)
        elif sentiment_label == -1:
            agent = ES_Agent_Negative(self.llm_35)
        else:
            return {"error": "Unexpected sentiment label"}

        result = agent._run_agent(input_text)
        if not result or result.strip() == "":
            self.logger.error(f"Agent returned an empty result: {result}")
            return {"error": "Agent returned an empty result"}

        try:
            if isinstance(result, dict):
                return result

            if isinstance(result, str):
                data = json.loads(result)
                return data

        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding intention result: {result}, Error: {e}")
            return {"error": "Failed to determine the intention due to invalid JSON."}
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return {"error": "Failed to determine the intention due to an unexpected error."}

# Intention dictionary to map intention codes to human-readable intentions
INTENTION_DICT = {
    # Positive Sentiment Narratives (ESP)
    "ESP1": "The Government’s Firm Approach Restoring Law and Order",
    "ESP2": "El Salvador is Becoming a Model of Security and Strong Leadership in Fighting Gang Violence",
    "ESP3": "Government Policies Attracting Support and International Investment",
    "ESP4": "El Salvador at the Forefront of Economic Modernization and Cryptocurrency Adoption",
    "ESP5": "Bitcoin Policy Opening Doors for Innovation and Attracting Entrepreneurs and International Investors",
    "ESP6": "The Bitcoin Revolution Could Break the Country’s Dependence on Traditional Financial Systems",
    "ESP7": "El Salvador Affirms Sovereignty and Emerges as a Regional Leader by Standing up to Foreign Pressure",
    "ESP8": "Independent Policies Lead to a New Era of Self-Sufficiency and Greater Regional Influence",
    "ESP9": "Alternative Media Providing Truth Suppressed by Traditional Media",
    "ESP10": "Government Promoting Transparency Through its Own Media Channels",
    "ESP11": "Economic Growth Promised by Government Reforms and Innovation",
    "ESP12": "Government Promoting Equality Policies and Empowering Women",
    "ESP13": "Support Programs for Vulnerable Women Reinforcing Commitment to Gender Equality",
    "ESP14": "Policies for LGBT Inclusion and Respect for Rights",
    "ESP15": "Government Promoting Tolerance and Combatting Discrimination",
    "ESP16": "Strengthening Institutions and Creating Transparency Mechanisms to Combat Corruption",
    "ESP17": "Public Support for Anti-Corruption Initiatives Increasing Trust in Institutions",

    # Negative Sentiment Narratives (ESN)
    "ESN1": "The Government’s Rigid Approach is Compromising Civil Liberties",
    "ESN2": "Security Policies Lean Towards Authoritarianism, Risking Escalation of Violence",
    "ESN3": "Government Actions Attracting International Scrutiny, Deterring Investment",
    "ESN4": "Cryptocurrency Policies are Risky, Endangering Economic Stability",
    "ESN5": "Bitcoin Policy Isolating the Country from Traditional Financial Allies",
    "ESN6": "Push for Bitcoin Deepens Economic Vulnerability",
    "ESN7": "El Salvador’s Distancing from International Partners is Alienating Allies",
    "ESN8": "Independent Policies Increase Isolation from Key Support Networks",
    "ESN9": "Alternative Media is Promoting Polarizing Narratives",
    "ESN10": "Government-Controlled Media Reduces Transparency",
    "ESN11": "Government Reforms are Creating Economic Uncertainty",
    "ESN12": "Insufficient Gender Policies Allow Discrimination Against Women to Persist",
    "ESN13": "Lack of Funding Weakens Support Programs for Women’s Rights",
    "ESN14": "Lack of Legal Protection for the LGBT Community Increases Vulnerability",
    "ESN15": "Government Criticized for Failing to Actively Promote LGBT Rights",
    "ESN16": "Corruption Allegations Weaken Public Trust",
    "ESN17": "Lack of Transparency Raises Concerns About Government Integrity"
}