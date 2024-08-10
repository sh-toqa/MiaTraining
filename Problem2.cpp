#include <iostream>
using namespace std;

int findIndex(int *numbs, int numbsSize, int target)
{
    // linear search
    for (int i = 0; i < numbsSize; i++)
        if (numbs[i] == target)
            return i;
    return -1;
}
int main()
{
    // Prompt for size of the array
    int size;
    // cout << "Enter number of elements: ";
    do
    {
        cin >> size;
    } while (size <= 0);

    // Fill the array
    // cout << "Enter " << size << " integer(s): ";
    int arr[size];
    for (int i = 0; i < size; i++)
        cin >> arr[i];

    // Prompt for target
    int target;
    // cout << "Enter target: ";
    cin >> target;

    // Call function to find index
    int index = findIndex(arr, size, target);

    // Print output
    cout << index << endl;
    return 0;
}