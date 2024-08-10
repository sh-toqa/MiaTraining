#include <iostream>
using namespace std;

int main()
{
    int n; // height
    // cout << "Enter height of pyramid: ";
    // input validation: number is non-negative
    do
    {
        cin >> n;
    } while (n < 0);

    int i = 1;
    while (i <= n)
    {
        for (int j = 1; j <= i; j++)
            cout << "*";
        cout << endl;
        i++;
    }
    return 0;
}
