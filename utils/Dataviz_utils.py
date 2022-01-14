import pandas as pd
import numpy as np
import folium
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns


def country_map(data, country, value, location, zoom, template, legend, threshold_scale):
    """
    Plots a world map with colored countries according to a variable value
    
    Parameters:
    data: DataFrame
        dataset
        
    country: string
        name of the variable containing countries
        
    value: int
        name of the variable to color countries
        
    location: [x,y]
        map center
        
    zoom: float
        map zoom
        
    template: .json file
        map template to delimiter countries
        
    legend: string
        legend name
        
    threshold_scale: list
        value thresholds for countries colors
        
    """
    if threshold_scale == None:
        # create a numpy array of length 6 and has linear spacing from the minium value to the maximum
        threshold_scale = np.linspace(data[value].min(),
                                      data[value].max(),
                                      6, dtype=int)
        threshold_scale = threshold_scale.tolist() # change the numpy array to a list
        threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum

    # let Folium determine the scale.
    world_map = folium.Map(location=location, 
                           zoom_start=zoom, min_zoom=zoom, max_zoom=zoom,
                           width=1000,height=600,
                           tiles='Stamen Terrain', max_bounds=False)

    # build the map
    world_map.choropleth(
        geo_data=template,
        data=data,
        columns=[country, value],
        key_on='feature.properties.name',
        threshold_scale=threshold_scale,
        fill_color='PuBuGn', # https://github.com/python-visualization/folium/blob/v0.2.0/folium/utilities.py#L104
        fill_opacity=0.7, 
        line_opacity=0.2,
        legend_name=legend,
        reset=True
    )
    
    display(world_map)
    
####################################

def wordcloud_region(df, region, value, max_words):
    """
    Plots a wordclound according to a variable value for a selected region
    
    Parameters
    ----------
    df : DataFrame
        dataset
     
    region: string
        selected region name
        
    value: string
        variable name
        
    max_words: ?
        ?????
    """
    #
    df_temp = df.loc[df['region']==region]
    df_temp.set_index("country", inplace=True)
    
    #
    total_happiness = df_temp[value].sum()
    
    #
    word_string = ''
    for country in df_temp.index.values:
    # check if country's name is a single-word name
        if len(country.split(' ')) == 1:
            repeat_num_times = int(df_temp.loc[country, value]/float(total_happiness)*max_words)
            word_string = word_string + ((country + ' ') * repeat_num_times)
    wordcloud = WordCloud(background_color='white').generate(word_string)
    
    return wordcloud

###

def multiple_wordcloud_region(df, region_list ,value, max_words ):
    """
    Plots a wordclound according to a variable value for several regions
    
    Parameters
    ----------
    df : DataFrame
        dataset
     
    region: list
        selected regions names
        
    value: string
        variable name
        
    max_words: ?
        ?????
    """
    # display the cloud
    fig = plt.figure(figsize=(20,5))
    j = 100+len(region_list)*10+1
    
    for r in region_list:
        wordcloud = wordcloud_region(df, r, value, max_words)
        plt.subplot(j)
        plt.imshow(wordcloud, interpolation='bilinear')
        j+=1
        plt.axis('off')
        plt.title(r, y=-0.3, fontsize=30)
    plt.show()


########################################
def bar_reg(df, by, value, title):
    """
    Plots variable values for each categories of a categorical variable
    
    Adds the frequency of the categories in the dataset
    
    df: DataFrame
        dataset
        
    by: string
        categorical variable name
        
    value: string
        variable
        
    title: string
        graph title
    """
    df_tmp = df.groupby(by, axis=0).mean().sort_values(value, ascending=False)
    df_tmp.reset_index(inplace=True)
    df_tmp.rename(columns={"index":by},inplace=True)

    s = df[by].value_counts().to_frame().rename(columns={by:"nb_countries"})
    d = dict(zip(s.index.values,s["nb_countries"].tolist()))

    norm = plt.Normalize(df_tmp[value].min(), df_tmp[value].max())
    cmap = plt.get_cmap("PuBuGn")

    g = sns.barplot(y=by, x=value, palette=cmap(norm(df_tmp[value].values)), data=df_tmp)
    ylabels = [r+'   ('+str(d[r])+')' for r in df_tmp["region"].tolist()]
    g.set_yticklabels(ylabels)
    g.set(ylabel='region (number of countries)')
    g.set_title(title)

    plt.show()
    
    
########################################
def multiple_bar_chart(df, catvar, numvar_list, palette):
    """
    for a selection of numerical variables: 
    Plots variable values for each categories of a categorical variable 
    
    df: DataFrame
        dataset
        
    catvar: string
        categorical variable name
        
    numvar_lsit: list
        list containing numerical variables names
        
    palette: list
        color palette
    
    """
    fig= plt.figure(figsize=(20,3))
    plt.style.use('ggplot')
    
    i = 0
    j = 161
    
    for l in numvar_list:
        df_temp = df[[catvar,l]].groupby(catvar, axis=0).mean().sort_values(l, ascending=True)
        df_temp.reset_index(inplace=True)
    
        plt.subplot(j)
        plt.bar(catvar, l, width=0.8, bottom=None, data=df_temp.sort_values(by=catvar), color=palette[i])
        plt.xticks(rotation=20)
        plt.title(l, y=-0.3)
        
        j+=1
        i+=1
    
    plt.show()
    
    
############################################
def numeric_th_graph(df, catvar, numvar, th, over_under="over"):
    """
    
    """
    plt.figure(figsize=(20,5))
    
    plt.subplot(121)
    plot = sns.scatterplot(y="happiness_score", x=numvar, data=df, hue=catvar, 
                           palette= ['#F6EFF7', '#A6BDDB', '#3690C0', '#016450'])
    plot.set_title("Distribution according to "+numvar+" and happiness_score", y=1.1)
    plot.axvline(th,color='r')
    plt.legend(loc='lower right')

    plt.subplot(122)
    if over_under=="over":
        df_high_corr = df[df[numvar]>th]
    elif over_under=="under":
        df_high_corr = df[df[numvar]<th]
    
    s = df_high_corr[catvar].value_counts(-1).rename(numvar).reset_index().sort_values("index", ascending=False)
    s.rename(columns={"index":"happiness",numvar:" "},inplace=True)
    
    
    s[" "].plot(kind='pie',
                                   figsize=(15, 6),
                                   autopct='%1.1f%%', 
                                   startangle=0,    
                                   shadow=False,       
                                   labels=None,         # turn off labels on pie chart
                                   pctdistance=1.12,    # the ratio between the center of each pie slice and the start of the text generated by autopct 
                                   colors=['#016450', '#3690C0', '#A6BDDB', '#F6EFF7'],  # add custom colors
                                   explode=[0.1, 0, 0, 0] # 'explode' lowest 3 continents
                                  )

    # scale the title up by 12% to match pctdistance
    if over_under=="over":
        plt.title('Happiness level for countries with '+numvar+' >'+str(th), y=1.1)
    elif over_under=="under":
        plt.title('Happiness level for countries with '+numvar+' <'+str(th), y=1.1)
    plt.axis('equal') 

    # add legend
    plt.legend(labels=s.happiness, loc='upper right') 

    plt.show()