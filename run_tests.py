import unittest
import sys
import os
import time
import test_app

def run_tests():
    # Get all test methods from our test class
    test_class = test_app.AzureUsageAnalysisTests
    test_methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    print(f"Found {len(test_methods)} test methods to run:")
    for method in test_methods:
        print(f"  - {method}")
    print()
    
    # Run each test individually
    results = []
    for test_method in test_methods:
        print(f"Running test: {test_method}")
        
        # Create a test suite with just this test
        suite = unittest.TestSuite()
        suite.addTest(test_class(test_method))
        
        # Run the test
        runner = unittest.TextTestRunner()
        result = runner.run(suite)
        
        # Store the result
        success = result.wasSuccessful()
        results.append((test_method, success))
        
        print(f"Test {'PASSED' if success else 'FAILED'}")
        print("-" * 40)
        
        # Small delay to allow output to be displayed properly
        time.sleep(0.5)
    
    # Print summary
    print("\nTest Results Summary:")
    print("=" * 40)
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    for test_method, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_method}")
    
    print("-" * 40)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    # Return success if all tests passed
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 