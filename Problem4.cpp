#include <iostream>
using namespace std;

void trackScore(int **justice_league, int **villains, int r, int c, int *jl_counter, int *v_counter)
{
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
            if (justice_league[i][j] > villains[i][j])
                (*jl_counter)++;
            else if (villains[i][j] > justice_league[i][j])
                (*v_counter)++;
    }
}
int main()
{
    // Set up the grid
    int r, c;
    cin >> r >> c;

    // Justice League power level
    int justice_league[r][c];
    int *ptr_jl[r];
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
            cin >> justice_league[i][j];
    }
    for (int i = 0; i < r; i++)
        ptr_jl[i] = justice_league[i];

    // Villans power level
    int villains[r][c];
    int *ptr_v[r];
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
            cin >> villains[i][j];
    }
    for (int i = 0; i < r; i++)
        ptr_v[i] = villains[i];

    // Track score of winners
    int justice_league_counter = 0;
    int villains_counter = 0;
    trackScore(ptr_jl, ptr_v, r, c, &justice_league_counter, &villains_counter);

    // Output
    if (justice_league_counter > villains_counter)
        cout << "Justice League" << endl;
    else if (villains_counter > justice_league_counter)
        cout << "Villains" << endl;
    else
        cout << "Tie" << endl;

    return 0;
}