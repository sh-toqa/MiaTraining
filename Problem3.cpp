#include <iostream>
using namespace std;

int main()
{
    // Prompt for the number of height measurements
    // cout << "Enter number of measurements: ";
    int n;
    do
    {
        cin >> n;
    } while (n <= 0);

    // Take measurements --> array
    // cout << "Enter " << n << " measurement(s): ";
    int measurements[n];
    for (int i = 0; i < n; i++)
        cin >> measurements[i];

    // Loop through array to find maximum
    int max = measurements[0];
    for (int i = 1; i < n; i++)
        if (measurements[i] > max)
            max = measurements[i];
    // Output
    cout << max << endl;
    return 0;
}
