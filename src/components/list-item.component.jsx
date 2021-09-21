const ListItem = (props) => {

    const handleDelete = (e) => {

    }

    return (
        <div className="mt-1 ml-1 p-1 max-w-full flex items-center bg-blue-700 text-white rounded select-none cursor-text">
            <span className="text-sm truncate ml-1">
                {props.item.name} 
                <span className="text-sm font-medium truncate ml-1">
                    {props.item.percent}%
                </span>
            </span>
            <button onClick={handleDelete}
                className="pl-1 flex items-center border-0 outline-none focus:outline-none focus:ring-0">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
            </button>
        </div>
    )
    
}

export default ListItem