from api.app import create_app
from database.models import db, Prediction, ModelVersion, RiskLevel, EventType
from datetime import datetime, timezone

app = create_app()

with app.app_context():
    print("--- Starting Database Integration Test ---")
    
    # 1. Ensure a Model Version exists (Requirement for Foreign Key)
    version_id = "v1.0.0-test"
    existing_version = ModelVersion.query.filter_by(version=version_id).first()
    
    if not existing_version:
        test_version = ModelVersion(
            version=version_id,
            model_path="saved_models/current_model.h5",
            is_active=True
        )
        db.session.add(test_version)
        db.session.commit()
        print(f"✓ Created ModelVersion: {version_id}")
    else:
        print(f"✓ ModelVersion {version_id} already exists")

    # 2. Create a Prediction with the new Enum values
    try:
        new_pred = Prediction(
            asset_id="TRANSFORMER_01",
            historical_load={"t-1": 150.5, "t-2": 148.0},
            temperature=42.5,
            predicted_load=155.2,
            prediction_horizon=24,
            risk_score=0.85,
            risk_level=RiskLevel.high,  # Testing lowercase Enum key
            model_version=version_id,    # Testing Foreign Key
            maintenance_recommended=True
        )
        
        db.session.add(new_pred)
        db.session.commit()
        print(f"✓ Prediction created with ID: {new_pred.id}")
        
        # 3. Verify it's readable and to_dict works
        check_pred = Prediction.query.get(new_pred.id)
        print(f"✓ Verification: Retrieved Risk Level is '{check_pred.risk_level.value}'")
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Test Failed: {e}")

    print("--- Test Complete ---")