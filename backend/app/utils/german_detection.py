# app/utils/german_detection.py
import re
from typing import List, Tuple, Dict

class GermanComplianceDetector:
    """Detect German compliance documents and terminology"""
    
    GERMAN_LEGAL_TERMS = {
        # GDPR/DSGVO Terms
        "dsgvo": ["DSGVO", "Datenschutzgrundverordnung"],
        "personal_data": ["personenbezogene Daten", "Personendaten"],
        "processing": ["Verarbeitung", "verarbeitet", "verarbeiten"],
        "consent": ["Einwilligung", "einwilligen", "Zustimmung"],
        "ropa": ["Verfahrensverzeichnis", "Verzeichnis von Verarbeitungstätigkeiten"],
        "dpia": ["Datenschutz-Folgenabschätzung", "DSFA"],
        "authority": ["Aufsichtsbehörde", "Datenschutzbehörde", "BfDI", "BayLDA", "LfDI"],
        "rights": ["Betroffenenrechte", "Auskunftsrecht", "Löschungsrecht"],
        "legal_basis": ["Rechtsgrundlage", "Art. 6", "Artikel 6"],
        "privacy_policy": ["Datenschutzerklärung", "Datenschutzrichtlinie"],
        "security": ["Sicherheitsmaßnahmen", "technische Maßnahmen", "organisatorische Maßnahmen"]
    }
    
    GDPR_ARTICLES_DE = {
        "Art. 5": "Grundsätze für die Verarbeitung",
        "Art. 6": "Rechtmäßigkeit der Verarbeitung", 
        "Art. 7": "Bedingungen für die Einwilligung",
        "Art. 13": "Informationspflicht bei Erhebung",
        "Art. 14": "Informationspflicht bei Dritterhebung",
        "Art. 15": "Auskunftsrecht der betroffenen Person",
        "Art. 16": "Recht auf Berichtigung",
        "Art. 17": "Recht auf Löschung",
        "Art. 18": "Recht auf Einschränkung der Verarbeitung",
        "Art. 20": "Recht auf Datenübertragbarkeit",
        "Art. 25": "Datenschutz durch Technikgestaltung",
        "Art. 30": "Verzeichnis von Verarbeitungstätigkeiten",
        "Art. 32": "Sicherheit der Verarbeitung",
        "Art. 35": "Datenschutz-Folgenabschätzung"
    }
    
    @classmethod
    def detect_language(cls, content: str, filename: str = "") -> Tuple[str, float]:
        """Detect if document is German with confidence score"""
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        german_score = 0
        total_checks = 0
        
        # Check filename for German indicators
        german_filename_indicators = [
            'datenschutz', 'dsgvo', 'verfahrensverzeichnis', 'dsfa',
            'sicherheit', 'richtlinie', 'verarbeitung'
        ]
        
        for indicator in german_filename_indicators:
            total_checks += 1
            if indicator in filename_lower:
                german_score += 3  # Weight filename heavily
        
        # Check content for German terms
        all_german_terms = []
        for category, terms in cls.GERMAN_LEGAL_TERMS.items():
            all_german_terms.extend([term.lower() for term in terms])
        
        for term in all_german_terms:
            total_checks += 1
            if term in content_lower:
                german_score += 1
        
        # Check for GDPR articles in German format
        for article in cls.GDPR_ARTICLES_DE.keys():
            total_checks += 1
            if article.lower() in content_lower:
                german_score += 2
        
        if total_checks == 0:
            return "unknown", 0.0
        
        confidence = german_score / total_checks
        
        if confidence > 0.3:
            return "de", confidence
        elif confidence > 0.1:
            return "mixed", confidence
        else:
            return "en", 1.0 - confidence
    
    @classmethod
    def extract_german_terms(cls, content: str) -> Dict[str, List[str]]:
        """Extract German legal terms found in content"""
        found_terms = {}
        content_lower = content.lower()
        
        for category, terms in cls.GERMAN_LEGAL_TERMS.items():
            found_in_category = []
            for term in terms:
                if term.lower() in content_lower:
                    found_in_category.append(term)
            
            if found_in_category:
                found_terms[category] = found_in_category
        
        return found_terms
    
    @classmethod
    def extract_gdpr_articles(cls, content: str) -> List[str]:
        """Extract GDPR article references from content"""
        articles_found = []
        content_lower = content.lower()
        
        for article in cls.GDPR_ARTICLES_DE.keys():
            if article.lower() in content_lower:
                articles_found.append(article)
        
        # Also check for variations like "Artikel 6" instead of "Art. 6"
        article_patterns = [
            r'artikel\s+(\d+)',
            r'art\.\s*(\d+)',
            r'art\s+(\d+)'
        ]
        
        for pattern in article_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                article_ref = f"Art. {match}"
                if article_ref not in articles_found and article_ref in cls.GDPR_ARTICLES_DE:
                    articles_found.append(article_ref)
        
        return articles_found