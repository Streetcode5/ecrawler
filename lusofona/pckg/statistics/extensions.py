
#import pandas as pd 

# Adds variance, skewness and kurtosis to the describe() method
def extended_describe( df ):
    numerical = df.select_dtypes( exclude=[ 'object', 'category', 'datetime64[ns]' ] )
    extended_df = df.describe( include='all' ).T
    extended_df['var'] = numerical.var()
    extended_df['skew'] = numerical.skew()
    extended_df['kurt'] = numerical.kurt()
    return extended_df