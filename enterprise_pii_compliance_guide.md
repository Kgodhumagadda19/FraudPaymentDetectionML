# Enterprise PII Compliance Guide for Fraud Detection

## 🛡️ **PRIVACY-FIRST FRAUD DETECTION**

This guide provides comprehensive strategies for using PII data in fraud detection while maintaining full compliance with privacy regulations.

---

## 📋 **TABLE OF CONTENTS**

1. [Legal Framework](#legal-framework)
2. [Privacy-Preserving Techniques](#privacy-preserving-techniques)
3. [Synthetic PII Training](#synthetic-pii-training)
4. [Anonymization Methods](#anonymization-methods)
5. [Federated Learning](#federated-learning)
6. [Enterprise Implementation](#enterprise-implementation)
7. [Compliance Checklist](#compliance-checklist)

---

## ⚖️ **LEGAL FRAMEWORK**

### **GDPR (General Data Protection Regulation)**
- **Article 6**: Legal basis for processing
- **Article 25**: Privacy by design and default
- **Article 32**: Security of processing
- **Article 35**: Data protection impact assessment

### **CCPA (California Consumer Privacy Act)**
- **Section 1798.100**: Right to know
- **Section 1798.105**: Right to delete
- **Section 1798.120**: Right to opt-out

### **PCI DSS (Payment Card Industry Data Security Standard)**
- **Requirement 3**: Protect stored cardholder data
- **Requirement 4**: Encrypt transmission of cardholder data
- **Requirement 7**: Restrict access to cardholder data

---

## 🔐 **PRIVACY-PRESERVING TECHNIQUES**

### **1. Data Minimization**
```python
# Only collect necessary PII
required_pii = ['email', 'phone', 'ip_address']
optional_pii = ['name', 'address', 'ssn']

# Use minimal PII for fraud detection
def minimize_pii_data(transaction_data):
    return {
        'email_hash': hash_email(transaction_data['email']),
        'phone_hash': hash_phone(transaction_data['phone']),
        'ip_network': extract_network(transaction_data['ip_address'])
    }
```

### **2. Pseudonymization**
```python
# Replace identifiers with pseudonyms
def pseudonymize_data(data):
    return {
        'customer_id': generate_pseudonym(data['email']),
        'device_id': generate_pseudonym(data['device_id']),
        'session_id': generate_pseudonym(data['session_id'])
    }
```

### **3. Encryption at Rest and in Transit**
```python
# Encrypt sensitive data
from cryptography.fernet import Fernet

def encrypt_pii_data(data, key):
    f = Fernet(key)
    encrypted_data = {}
    for field, value in data.items():
        if is_sensitive(field):
            encrypted_data[field] = f.encrypt(value.encode())
    return encrypted_data
```

---

## 🤖 **SYNTHETIC PII TRAINING**

### **Benefits of Synthetic PII**
- ✅ No real customer data
- ✅ No privacy violations
- ✅ Unlimited training data
- ✅ Legal compliance
- ✅ Risk-free development

### **Implementation Strategy**
```python
# Generate synthetic PII for training
def generate_synthetic_pii_dataset(n_samples=100000):
    return {
        'email': [f"user{i}@example.com" for i in range(n_samples)],
        'phone': [f"+1-555-{i:04d}" for i in range(n_samples)],
        'ip_address': [f"192.168.{i//256}.{i%256}" for i in range(n_samples)],
        'credit_card': [f"****-****-****-{i:04d}" for i in range(n_samples)],
        'address': [f"123 Main St, City {i}" for i in range(n_samples)]
    }
```

---

## 🔒 **ANONYMIZATION METHODS**

### **1. Hashing Techniques**
```python
import hashlib
import hmac

def secure_hash(data, salt):
    """Create secure hash with salt"""
    return hmac.new(salt.encode(), data.encode(), hashlib.sha256).hexdigest()

def hash_email(email):
    """Hash email address"""
    return secure_hash(email.lower(), "email_salt")[:16]

def hash_phone(phone):
    """Hash phone number"""
    return secure_hash(re.sub(r'\D', '', phone), "phone_salt")[:16]
```

### **2. Tokenization**
```python
def tokenize_credit_card(card_number):
    """Tokenize credit card number"""
    return f"tok_{hash(card_number) % 1000000:06d}"

def tokenize_ssn(ssn):
    """Tokenize SSN"""
    return f"ssn_{hash(ssn) % 1000000:06d}"
```

### **3. Feature Extraction**
```python
def extract_ip_features(ip_address):
    """Extract features from IP without storing full IP"""
    parts = ip_address.split('.')
    return {
        'ip_country': get_country_from_ip(ip_address),
        'ip_network': f"{parts[0]}.{parts[1]}.*.*",
        'ip_risk_score': calculate_ip_risk(ip_address)
    }
```

---

## 🌐 **FEDERATED LEARNING**

### **Concept**
Train models on local data without sharing raw PII across systems.

### **Implementation**
```python
class FederatedFraudDetection:
    def __init__(self):
        self.global_model = None
        self.local_models = {}
    
    def train_local_model(self, local_data, client_id):
        """Train model on local data"""
        # Train model locally
        local_model = train_model(local_data)
        
        # Only share model weights, not data
        return local_model.get_weights()
    
    def aggregate_models(self, local_weights):
        """Aggregate model weights from all clients"""
        # Combine weights without sharing data
        return aggregate_weights(local_weights)
    
    def update_global_model(self, aggregated_weights):
        """Update global model with aggregated weights"""
        self.global_model.set_weights(aggregated_weights)
```

---

## 🏢 **ENTERPRISE IMPLEMENTATION**

### **1. Data Governance Framework**
```python
class PIIGovernance:
    def __init__(self):
        self.data_classification = {
            'high_sensitivity': ['ssn', 'credit_card', 'passport'],
            'medium_sensitivity': ['email', 'phone', 'address'],
            'low_sensitivity': ['ip_address', 'user_agent']
        }
    
    def classify_data(self, field_name):
        """Classify data sensitivity level"""
        for level, fields in self.data_classification.items():
            if field_name in fields:
                return level
        return 'unknown'
    
    def apply_controls(self, data, classification):
        """Apply appropriate controls based on classification"""
        if classification == 'high_sensitivity':
            return self.encrypt_data(data)
        elif classification == 'medium_sensitivity':
            return self.hash_data(data)
        else:
            return self.anonymize_data(data)
```

### **2. Consent Management**
```python
class ConsentManager:
    def __init__(self):
        self.consent_records = {}
    
    def record_consent(self, user_id, purpose, consent_given):
        """Record user consent for data processing"""
        self.consent_records[user_id] = {
            'purpose': purpose,
            'consent_given': consent_given,
            'timestamp': datetime.now(),
            'expiry': datetime.now() + timedelta(years=1)
        }
    
    def check_consent(self, user_id, purpose):
        """Check if user has given consent for specific purpose"""
        if user_id not in self.consent_records:
            return False
        
        record = self.consent_records[user_id]
        return (record['purpose'] == purpose and 
                record['consent_given'] and 
                record['expiry'] > datetime.now())
```

### **3. Data Retention Policies**
```python
class DataRetention:
    def __init__(self):
        self.retention_policies = {
            'fraud_detection': timedelta(days=90),
            'compliance': timedelta(years=7),
            'analytics': timedelta(days=365)
        }
    
    def should_retain(self, data_type, creation_date):
        """Check if data should be retained"""
        policy = self.retention_policies.get(data_type)
        if not policy:
            return False
        
        return datetime.now() - creation_date < policy
    
    def cleanup_expired_data(self):
        """Remove expired data"""
        for data_type, policy in self.retention_policies.items():
            cutoff_date = datetime.now() - policy
            # Remove data older than cutoff_date
            pass
```

---

## ✅ **COMPLIANCE CHECKLIST**

### **Pre-Implementation**
- [ ] **Data Protection Impact Assessment (DPIA)**
- [ ] **Legal basis for processing identified**
- [ ] **Consent mechanisms implemented**
- [ ] **Data minimization strategy defined**
- [ ] **Security controls designed**

### **Implementation**
- [ ] **PII data encrypted at rest**
- [ ] **PII data encrypted in transit**
- [ ] **Access controls implemented**
- [ ] **Audit logging enabled**
- [ ] **Data retention policies applied**

### **Post-Implementation**
- [ ] **Regular security audits**
- [ ] **Privacy training for staff**
- [ ] **Incident response plan tested**
- [ ] **Compliance monitoring active**
- [ ] **Regular policy reviews**

---

## 🚀 **BEST PRACTICES**

### **1. Privacy by Design**
- Embed privacy controls from the start
- Use synthetic data for development
- Implement data minimization
- Enable user control over data

### **2. Security First**
- Encrypt all sensitive data
- Use secure communication channels
- Implement access controls
- Regular security assessments

### **3. Transparency**
- Clear privacy policies
- User consent mechanisms
- Data usage notifications
- Right to access and delete

### **4. Continuous Monitoring**
- Regular compliance audits
- Privacy impact assessments
- Incident monitoring
- Policy updates

---

## 📞 **RESOURCES**

### **Legal Resources**
- [GDPR Official Text](https://gdpr.eu/)
- [CCPA Official Text](https://oag.ca.gov/privacy/ccpa)
- [PCI DSS Requirements](https://www.pcisecuritystandards.org/)

### **Technical Resources**
- [OWASP Privacy Risks](https://owasp.org/www-project-privacy-risks/)
- [NIST Privacy Framework](https://www.nist.gov/privacy-framework)
- [ISO 27701 Privacy Information Management](https://www.iso.org/iso-27701-privacy-information-management.html)

---

## ⚠️ **DISCLAIMER**

This guide provides general information about PII compliance in fraud detection. It is not legal advice. Organizations should consult with legal counsel to ensure compliance with applicable laws and regulations in their jurisdiction.

**Last Updated:** July 2024
**Version:** 1.0 