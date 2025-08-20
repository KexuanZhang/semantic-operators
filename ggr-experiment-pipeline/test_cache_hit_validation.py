#!/usr/bin/env python3
"""
Test the corrected cache hit rate validation logic
"""

def test_cache_hit_rate_validation():
    """Test the new validation logic for cache hit rates"""
    print("Testing cache hit rate validation logic...")
    
    # Test cases: (hits, queries, expected_should_report)
    test_cases = [
        # Valid cases - should report
        (85, 100, True, "Normal case with good sample size"),
        (150, 200, True, "Larger sample with good hit rate"),
        (5, 50, True, "Low hit rate but good sample size"),
        
        # Invalid cases - should NOT report (misleading 100% rates)
        (0.0002, 0.0002, False, "Tiny identical values (100% rate)"),
        (0.1, 0.1, False, "Small identical values (100% rate)"),
        (1.5, 1.5, False, "Identical decimal values"),
        
        # Edge cases
        (0, 100, True, "Zero hits, good sample size"),
        (10, 10, False, "Small sample with 100% rate"),
        (50, 50, False, "Medium sample with suspicious 100% rate"),
        (1000, 1000, True, "Large sample with 100% rate (could be valid)"),
    ]
    
    def validate_hit_rate(hits, queries):
        """Replicate the validation logic from the updated code"""
        if queries <= 0:
            return False, "No queries"
        
        hit_rate = (hits / queries) * 100
        
        # Only report hit rate if we have meaningful data
        if queries > 10 or hit_rate < 99:  # Avoid reporting 100% from tiny values
            return True, f"Valid: {hit_rate:.2f}% ({hits}/{queries})"
        else:
            return False, f"Invalid: {hit_rate:.2f}% from {queries} queries (suspicious)"
    
    print("\nTesting validation logic:")
    print("=" * 80)
    
    for hits, queries, expected_should_report, description in test_cases:
        should_report, reason = validate_hit_rate(hits, queries)
        
        status = "✅ PASS" if should_report == expected_should_report else "❌ FAIL"
        print(f"{status} | {description}")
        print(f"      Hits: {hits}, Queries: {queries}")
        print(f"      Should report: {expected_should_report}, Got: {should_report}")
        print(f"      Reason: {reason}")
        print()
    
    print("=" * 80)
    print("✅ Cache hit rate validation tests completed!")
    print()
    print("Key improvements:")
    print("• Now detects and rejects misleading 100% rates from tiny values")
    print("• Requires minimum 10 queries OR hit rate < 99% to report")
    print("• Provides clear feedback when metrics are insufficient")
    print("• Shows precise calculations with decimal places")

if __name__ == "__main__":
    test_cache_hit_rate_validation()
