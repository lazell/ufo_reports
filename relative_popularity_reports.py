
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Finds Relative Popular Events of  UFO reporting data
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''


def main():
    # Load report data and population data
    df_1947_usa = load_and_process('data/data.pkl')
    us_pop, internet_pop = US_populations()

    # Factor in population data and return charts of most reported
    most_reported_dates = list_of_popular_dates(df_1947_usa,
                                                internet_pop,
                                                99.6,
                                                'date')
    relative_most_reported = custom_filter_dates(most_reported_dates)

    # Print relative most reported
    print (relative_most_reported)
    # Display Charts
    plt.show()

def load_and_process(datafile):
    # Load data
    df = pd.read_pickle(datafile)

    # Clean data
    df_sorted = df.sort_values('occurred_clean')
    df.drop('occurred_len', axis=1, inplace=True)

    # Generate time and date columns
    df['Hour_of_day'] = [x.hour for x in df['occurred_clean']]
    df['date'] = [str(x.year)+'-'+str(x.month)+'-'+str(x.day) for x in df['occurred_clean']]
    df['month-year'] = [str(x.year) + '-' + str(x.month) for x in df['occurred_clean']]
    df['year'] = [x.year for x in df['occurred_clean']]

    # Filter for US entries only (based on valid state) after Roswell (1947)

    states = '''AL,AK,AS,AZ,AR,CA,CO,CT,DE,DC,FM,FL,GA,GU,HI,ID,IL,IN,IA,KS,KY,
                LA,ME,MH,MD,MA,MI,MN,MS,MO,MT,NE,NV,NH,NJ,NM,NY,NC,ND,MP,OH,OK,
                OR,PW,PA,PR,RI,SC,SD,TN,TX,UT,VT,VI,VA,WA,WV,WI,WY'''.split(',')

    df_1947_usa = df[(df['occurred_clean'] > '1947-06-24') & (df['state'].isin(states))]


    return df_1947_usa


def most_popular_reporting_time(df):
    # Sums the number of reports per reporting hour (0 = midnight, 23= 11.00pm)
    return pd.crosstab(df['date'], df['Hour_of_day']).sum()


def US_populations():

    # Generates dataframe of internet population by year, source:
    # Http://www.internetlivestats.com/internet-users/

    internet_pop = pd.read_table('data/Internet_users_by_year.txt', sep=',')
    internet_pop['Internet_Users'] = internet_pop['Internet_Users'].astype(int)
    internet_pop['year'] = internet_pop['Year'].astype(int)
    # Factorize by 1994 internet population:
    int_pop_1994 = 25437639
    internet_pop['factorized'] = internet_pop['Internet_Users']/int_pop_1994
    internet_pop.set_index('Year', inplace=True)

    # Generates dataframe of internet population by year, source:
    # Http://www.internetlivestats.com/internet-users

    us_pop = pd.read_table('data/us_population_year.csv', sep='\t')
    us_pop['year'] = us_pop['Date'].str[-4:].astype(int)
    us_pop['population (in millions)'] = us_pop['Value'].str[:6].astype(float)
    us_pop['factorized'] = us_pop['population (in millions)']/144.13
    us_pop.drop('Date', axis=1, inplace=True)
    us_pop.drop('Value', axis=1, inplace=True)

    return us_pop, internet_pop


def list_of_popular_dates(df, population, perc, type_='date'):

    # Create table of counts per month-year
    counts = pd.crosstab(df['state'], df[type_]).sum()
    df_counts = pd.DataFrame(counts, columns=['counts'])

    # Apply factorization to counts
    df_counts[type_] = df_counts.index
    df_counts['year'] = df_counts.index.astype(str).str[:4].astype(int)
    df_counts = df_counts.merge(population, on='year', how='outer')
    df_counts['factorized'].fillna(1, inplace=True)
    df_counts['relative counts'] = df_counts['counts']/(4+df_counts['factorized'])

    # Select relavent counts by percentile.
    threshold = np.percentile(df_counts['relative counts'], perc)
    most_reported = df_counts[df_counts['relative counts'] >= threshold]

    # Displat Chart
    plt.figure(figsize=(15, 6))
    plt.bar(most_reported[type_],
            most_reported['relative counts'],
            color='#5A3379')
    plt.title('Most reported UFO sightings (factoring in US internet population)')
    plt.xticks(rotation=70)


    # DataFrame of month-year dates

    most_reported_filter = list(most_reported[type_])
    return most_reported.sort_values('relative counts', ascending=False)

def custom_filter_dates(most_reported_dates):
    # Custom clean of relative most reported
    relative_most_reported = most_reported_dates[(most_reported_dates['date']
                                                 .str
                                                 .contains('-6-') == False)]
    relative_most_reported = relative_most_reported[(most_reported_dates['date']
                                                     .str
                                                     .contains('-7-15') == False)]
    # Display Chart
    plt.figure(figsize=(10, 6))
    plt.bar(relative_most_reported['date'],
            relative_most_reported['relative counts'],
            color='#336379', alpha=0.5)
    plt.title('Most Reported Events')
    plt.ylabel('Relative Popularity Score')
    plt.xticks(rotation=90)


    return relative_most_reported.sort_values(by='relative counts',
                                              ascending=False)

if __name__ == '__main__':
    main()
