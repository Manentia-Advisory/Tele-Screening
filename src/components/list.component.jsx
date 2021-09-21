import ListItem from "./list-item.component"

const List = (props) => (
    <div className="relative mx-12 mb-10">
        <h3 className="font-medium text-gray-900">Estimated Abnormalities List</h3>
            <div className="border border-gray-300 rounded">
                <div className="flex flex-wrap items-center p-1.5">
                {
                    props.list.map(i => (
                        <ListItem key={i.id} item={i} />      
                    ))
                }
                </div>
            </div>
        <p className="mt-0.5 text-sm text-gray-500">
            Please remove the diseases which are inaccurate (if any).
        </p>
    </div>
)

export default List